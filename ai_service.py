"""
AI Service for GDP Platform
Integrates OpenAI, Transformers, and custom AI models
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import json
from datetime import datetime, timedelta

import openai
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.config import settings
from core.models.gdp_models import NLQueryRequest, NLQueryResponse
from services.gdp_service import GDPService
from services.data_service import DataService
from core.ai_models.rag_system import RAGSystem
from core.ai_models.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class AIService:
    """
    Comprehensive AI service for GDP analytics platform
    """
    
    def __init__(self):
        # Initialize OpenAI
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
        
        # Initialize models
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        # Initialize services
        self.gdp_service = GDPService()
        self.data_service = DataService()
        self.rag_system = RAGSystem()
        self.knowledge_graph = KnowledgeGraph()
        
        # Cache for embeddings and responses
        self.embedding_cache = {}
        self.response_cache = {}
        
    async def process_natural_language_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        include_visualization: bool = True
    ) -> NLQueryResponse:
        """
        Process natural language queries about GDP data
        """
        try:
            logger.info(f"Processing NL query: {query}")
            
            start_time = datetime.utcnow()
            
            # Check cache first
            cache_key = f"{query}_{str(context)}"
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if (datetime.utcnow() - cached_response['timestamp']).seconds < 300:  # 5 min cache
                    return cached_response['response']
            
            # Parse query to understand intent
            intent = await self._parse_query_intent(query)
            
            # Extract entities (countries, time periods, indicators)
            entities = await self._extract_entities(query)
            
            # Retrieve relevant data
            relevant_data = await self._retrieve_relevant_data(entities, intent, context)
            
            # Generate response using RAG
            response_text = await self._generate_response(query, relevant_data, intent)
            
            # Create visualizations if requested
            visualizations = []
            if include_visualization and relevant_data:
                visualizations = await self._generate_visualizations(
                    relevant_data, intent, entities
                )
            
            # Calculate confidence score
            confidence = await self._calculate_confidence(query, relevant_data, intent)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(query, entities, intent)
            
            # Get data sources
            sources = self._get_data_sources(relevant_data)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = NLQueryResponse(
                answer=response_text,
                data=relevant_data,
                visualizations=visualizations,
                confidence=confidence,
                sources=sources,
                suggestions=suggestions,
                processing_time=processing_time
            )
            
            # Cache response
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': datetime.utcnow()
            }
            
            logger.info(f"NL query processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing NL query: {str(e)}", exc_info=True)
            return NLQueryResponse(
                answer="I encountered an error while processing your query. Please try rephrasing or contact support.",
                confidence=0.0,
                sources=[],
                suggestions=["Try asking about specific countries or time periods"],
                processing_time=0.0
            )
    
    async def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse query to understand user intent"""
        
        # Use OpenAI for intent classification
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an AI assistant that classifies GDP-related queries into intents.
                        Possible intents: 
                        - gdp_calculation: User wants to calculate GDP
                        - gdp_comparison: User wants to compare GDP between countries/periods
                        - gdp_forecast: User wants GDP predictions
                        - gdp_trends: User wants to analyze GDP trends
                        - gdp_components: User wants component breakdown
                        - data_quality: User asks about data quality
                        - general_info: General information about GDP
                        
                        Return JSON with: {"intent": "intent_name", "confidence": 0.0-1.0, "parameters": {}}
                        """
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.1
            )
            
            intent_data = json.loads(response.choices[0].message.content)
            return intent_data
            
        except Exception as e:
            logger.error(f"Error parsing intent: {str(e)}")
            return {
                "intent": "general_info",
                "confidence": 0.5,
                "parameters": {}
            }
    
    async def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query using NER"""
        
        entities = {
            "countries": [],
            "time_periods": [],
            "indicators": [],
            "methods": []
        }
        
        # Use regex patterns for basic entity extraction
        import re
        
        # Country codes pattern
        country_pattern = r'\b[A-Z]{2,3}\b'
        countries = re.findall(country_pattern, query.upper())
        entities["countries"] = list(set(countries))
        
        # Time periods
        time_patterns = [
            r'\b20\d{2}\b',  # Years like 2024
            r'\b20\d{2}-Q[1-4]\b',  # Quarters like 2024-Q1
            r'\b20\d{2}-\d{2}\b'  # Months like 2024-03
        ]
        
        for pattern in time_patterns:
            periods = re.findall(pattern, query)
            entities["time_periods"].extend(periods)
        
        # GDP-related indicators
        indicator_keywords = [
            "gdp", "gross domestic product", "economic growth", "inflation",
            "unemployment", "consumption", "investment", "exports", "imports"
        ]
        
        query_lower = query.lower()
        for keyword in indicator_keywords:
            if keyword in query_lower:
                entities["indicators"].append(keyword)
        
        # Calculation methods
        method_keywords = ["expenditure", "income", "output", "production"]
        for method in method_keywords:
            if method in query_lower:
                entities["methods"].append(method)
        
        return entities
    
    async def _retrieve_relevant_data(
        self,
        entities: Dict[str, List[str]],
        intent: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Retrieve relevant data based on entities and intent"""
        
        relevant_data = {}
        
        try:
            # Get GDP data for mentioned countries
            if entities["countries"]:
                for country in entities["countries"]:
                    country_data = await self.gdp_service.get_historical_data(
                        country_code=country,
                        start_date="2020-01-01",
                        end_date=None,
                        include_components=True
                    )
                    relevant_data[f"gdp_data_{country}"] = country_data
            
            # Get forecasts if intent is forecast-related
            if intent["intent"] == "gdp_forecast" and entities["countries"]:
                for country in entities["countries"]:
                    forecast_data = await self.gdp_service.get_forecasts(
                        country_code=country,
                        horizon=8
                    )
                    relevant_data[f"forecast_data_{country}"] = forecast_data
            
            # Get comparison data if intent is comparison
            if intent["intent"] == "gdp_comparison" and len(entities["countries"]) > 1:
                comparison_data = await self.gdp_service.compare_countries(
                    country_codes=entities["countries"],
                    normalize=True
                )
                relevant_data["comparison_data"] = comparison_data
            
            # Add context data if available
            if context:
                relevant_data["context"] = context
                
        except Exception as e:
            logger.error(f"Error retrieving relevant data: {str(e)}")
        
        return relevant_data
    
    async def _generate_response(
        self,
        query: str,
        relevant_data: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> str:
        """Generate natural language response using RAG"""
        
        try:
            # Prepare context from relevant data
            context_text = self._prepare_context_text(relevant_data)
            
            # Use OpenAI for response generation
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are an expert economist and data analyst specializing in GDP analysis.
                        Provide clear, accurate, and insightful responses about GDP data.
                        Use the provided data context to answer questions.
                        Format numbers clearly and provide context for economic indicators.
                        Be concise but comprehensive in your explanations.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Question: {query}
                        
                        Available Data Context:
                        {context_text}
                        
                        Please provide a comprehensive answer based on the available data.
                        """
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I couldn't generate a response at this time. Please try again later."
    
    def _prepare_context_text(self, relevant_data: Dict[str, Any]) -> str:
        """Prepare context text from relevant data"""
        
        context_parts = []
        
        for key, data in relevant_data.items():
            if isinstance(data, dict):
                if "gdp_data" in key:
                    country = key.split("_")[-1]
                    context_parts.append(f"GDP data for {country}: {json.dumps(data, default=str)[:500]}...")
                elif "forecast_data" in key:
                    country = key.split("_")[-1]
                    context_parts.append(f"Forecast data for {country}: {json.dumps(data, default=str)[:500]}...")
                elif key == "comparison_data":
                    context_parts.append(f"Comparison data: {json.dumps(data, default=str)[:500]}...")
        
        return "\n\n".join(context_parts)
    
    async def _generate_visualizations(
        self,
        relevant_data: Dict[str, Any],
        intent: Dict[str, Any],
        entities: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Generate visualization configurations"""
        
        visualizations = []
        
        try:
            # Time series chart for GDP trends
            if any("gdp_data" in key for key in relevant_data.keys()):
                visualizations.append({
                    "type": "line_chart",
                    "title": "GDP Trends",
                    "data_key": "gdp_trends",
                    "config": {
                        "x_axis": "period",
                        "y_axis": "gdp_value",
                        "color_by": "country"
                    }
                })
            
            # Bar chart for country comparison
            if intent["intent"] == "gdp_comparison":
                visualizations.append({
                    "type": "bar_chart",
                    "title": "GDP Comparison",
                    "data_key": "comparison_data",
                    "config": {
                        "x_axis": "country",
                        "y_axis": "gdp_value",
                        "sort_by": "gdp_value"
                    }
                })
            
            # Map visualization for geographic data
            if len(entities["countries"]) > 2:
                visualizations.append({
                    "type": "choropleth_map",
                    "title": "GDP by Country",
                    "data_key": "geographic_data",
                    "config": {
                        "color_scale": "viridis",
                        "hover_data": ["gdp_value", "growth_rate"]
                    }
                })
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        return visualizations
    
    async def _calculate_confidence(
        self,
        query: str,
        relevant_data: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the response"""
        
        confidence_factors = []
        
        # Intent confidence
        confidence_factors.append(intent.get("confidence", 0.5))
        
        # Data availability confidence
        data_score = min(1.0, len(relevant_data) / 3.0)  # Higher confidence with more data
        confidence_factors.append(data_score)
        
        # Query clarity (simple heuristic based on length and question words)
        query_words = query.lower().split()
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        has_question_word = any(word in query_words for word in question_words)
        clarity_score = 0.8 if has_question_word else 0.6
        confidence_factors.append(clarity_score)
        
        # Return weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    async def _generate_suggestions(
        self,
        query: str,
        entities: Dict[str, List[str]],
        intent: Dict[str, Any]
    ) -> List[str]:
        """Generate related query suggestions"""
        
        suggestions = []
        
        # Based on intent
        if intent["intent"] == "gdp_calculation":
            suggestions.extend([
                "Compare GDP with neighboring countries",
                "Show GDP growth trends over time",
                "Analyze GDP components breakdown"
            ])
        elif intent["intent"] == "gdp_forecast":
            suggestions.extend([
                "Show forecast confidence intervals",
                "Compare different forecasting models",
                "Analyze forecast accuracy for this country"
            ])
        
        # Based on entities
        if entities["countries"]:
            for country in entities["countries"][:2]:  # Limit suggestions
                suggestions.append(f"Show economic indicators for {country}")
                suggestions.append(f"Compare {country} with regional peers")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _get_data_sources(self, relevant_data: Dict[str, Any]) -> List[str]:
        """Extract data sources from relevant data"""
        
        sources = set()
        
        for key, data in relevant_data.items():
            if isinstance(data, dict) and "metadata" in data:
                source = data["metadata"].get("source", "Unknown")
                sources.add(source)
        
        # Add default sources
        sources.update(["World Bank", "IMF", "OECD", "National Statistics Offices"])
        
        return list(sources)
    
    async def generate_insights(
        self,
        data: Dict[str, Any],
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered insights from GDP data"""
        
        insights = []
        
        try:
            # Trend analysis
            trend_insights = await self._analyze_trends(data, countries)
            insights.extend(trend_insights)
            
            # Anomaly detection insights
            anomaly_insights = await self._detect_anomalies_insights(data, countries)
            insights.extend(anomaly_insights)
            
            # Comparative insights
            if len(countries) > 1:
                comparative_insights = await self._generate_comparative_insights(data, countries)
                insights.extend(comparative_insights)
            
            # Predictive insights
            predictive_insights = await self._generate_predictive_insights(data, countries)
            insights.extend(predictive_insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    async def _analyze_trends(
        self,
        data: Dict[str, Any],
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze GDP trends and generate insights"""
        
        insights = []
        
        for country in countries:
            country_data = data.get(f"gdp_data_{country}", {})
            if not country_data:
                continue
            
            try:
                # Calculate growth rates
                gdp_values = [record["gdp_value"] for record in country_data.get("records", [])]
                if len(gdp_values) >= 2:
                    latest_growth = ((gdp_values[-1] - gdp_values[-2]) / gdp_values[-2]) * 100
                    
                    if latest_growth > 3:
                        insights.append({
                            "type": "trend",
                            "country": country,
                            "message": f"{country} shows strong GDP growth of {latest_growth:.1f}%",
                            "importance": "high",
                            "data_points": gdp_values[-4:]
                        })
                    elif latest_growth < -1:
                        insights.append({
                            "type": "trend",
                            "country": country,
                            "message": f"{country} shows GDP contraction of {abs(latest_growth):.1f}%",
                            "importance": "high",
                            "data_points": gdp_values[-4:]
                        })
            
            except Exception as e:
                logger.error(f"Error analyzing trends for {country}: {str(e)}")
        
        return insights
    
    async def _detect_anomalies_insights(
        self,
        data: Dict[str, Any],
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies and generate insights"""
        
        insights = []
        
        # Use simple statistical method for anomaly detection
        for country in countries:
            country_data = data.get(f"gdp_data_{country}", {})
            if not country_data:
                continue
            
            try:
                gdp_values = [record["gdp_value"] for record in country_data.get("records", [])]
                if len(gdp_values) >= 8:
                    mean_gdp = np.mean(gdp_values)
                    std_gdp = np.std(gdp_values)
                    
                    # Check latest value for anomaly
                    latest_value = gdp_values[-1]
                    z_score = abs((latest_value - mean_gdp) / std_gdp)
                    
                    if z_score > 2:  # 2 standard deviations
                        insights.append({
                            "type": "anomaly",
                            "country": country,
                            "message": f"Unusual GDP value detected for {country} (Z-score: {z_score:.2f})",
                            "importance": "medium",
                            "anomaly_score": z_score
                        })
            
            except Exception as e:
                logger.error(f"Error detecting anomalies for {country}: {str(e)}")
        
        return insights
    
    async def _generate_comparative_insights(
        self,
        data: Dict[str, Any],
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate comparative insights between countries"""
        
        insights = []
        
        try:
            # Compare latest GDP values
            country_gdps = {}
            for country in countries:
                country_data = data.get(f"gdp_data_{country}", {})
                if country_data and country_data.get("records"):
                    latest_gdp = country_data["records"][-1]["gdp_value"]
                    country_gdps[country] = latest_gdp
            
            if len(country_gdps) >= 2:
                sorted_countries = sorted(country_gdps.items(), key=lambda x: x[1], reverse=True)
                top_country = sorted_countries[0]
                bottom_country = sorted_countries[-1]
                
                ratio = top_country[1] / bottom_country[1]
                
                insights.append({
                    "type": "comparison",
                    "message": f"{top_country[0]} has {ratio:.1f}x larger GDP than {bottom_country[0]}",
                    "importance": "medium",
                    "countries": [top_country[0], bottom_country[0]],
                    "values": [top_country[1], bottom_country[1]]
                })
        
        except Exception as e:
            logger.error(f"Error generating comparative insights: {str(e)}")
        
        return insights
    
    async def _generate_predictive_insights(
        self,
        data: Dict[str, Any],
        countries: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate predictive insights using simple forecasting"""
        
        insights = []
        
        for country in countries:
            try:
                country_data = data.get(f"gdp_data_{country}", {})
                if not country_data:
                    continue
                
                gdp_values = [record["gdp_value"] for record in country_data.get("records", [])]
                if len(gdp_values) >= 4:
                    # Simple linear trend prediction
                    x = np.arange(len(gdp_values))
                    coeffs = np.polyfit(x, gdp_values, 1)
                    trend_slope = coeffs[0]
                    
                    # Predict next period
                    next_value = coeffs[0] * len(gdp_values) + coeffs[1]
                    current_value = gdp_values[-1]
                    predicted_growth = ((next_value - current_value) / current_value) * 100
                    
                    if abs(predicted_growth) > 0.5:  # Significant predicted change
                        direction = "growth" if predicted_growth > 0 else "decline"
                        insights.append({
                            "type": "prediction",
                            "country": country,
                            "message": f"Predicted {direction} of {abs(predicted_growth):.1f}% for {country} next period",
                            "importance": "low",
                            "predicted_value": next_value,
                            "confidence": 0.6  # Simple model, low confidence
                        })
            
            except Exception as e:
                logger.error(f"Error generating predictive insights for {country}: {str(e)}")
        
        return insights
    
    async def summarize_report(
        self,
        data: Dict[str, Any],
        report_type: str = "quarterly"
    ) -> str:
        """Generate executive summary using AI"""
        
        try:
            # Prepare data for summarization
            text_data = self._prepare_report_text(data, report_type)
            
            # Use summarization pipeline
            summary = self.summarization_pipeline(
                text_data,
                max_length=200,
                min_length=50,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing report: {str(e)}")
            return "Unable to generate summary at this time."
    
    def _prepare_report_text(self, data: Dict[str, Any], report_type: str) -> str:
        """Prepare text data for summarization"""
        
        text_parts = []
        
        text_parts.append(f"This {report_type} GDP report covers economic performance analysis.")
        
        # Add key data points
        for key, value in data.items():
            if isinstance(value, dict) and "gdp_value" in str(value):
                text_parts.append(f"GDP data shows economic indicators and trends.")
        
        return " ".join(text_parts)


# Factory function
def create_ai_service() -> AIService:
    """Create and return configured AI service"""
    return AIService()