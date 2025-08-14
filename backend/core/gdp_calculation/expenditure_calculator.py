"""
GDP Expenditure Approach Calculator
Implements C + I + G + (X - M) with AI enhancements
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from core.models.gdp_models import GDPCalculationResult, ExpenditureComponents
from core.ai_models.anomaly_detector import AnomalyDetector
from core.ai_models.imputation_model import AIImputer
from services.uncertainty_quantifier import UncertaintyQuantifier

logger = logging.getLogger(__name__)


@dataclass
class ExpenditureData:
    """Structure for expenditure approach data"""
    consumption: Optional[float] = None
    investment: Optional[float] = None
    government_spending: Optional[float] = None
    exports: Optional[float] = None
    imports: Optional[float] = None
    timestamp: datetime = None
    source: str = ""
    confidence_score: float = 1.0


class ExpenditureCalculator:
    """
    Advanced GDP Calculator using Expenditure Approach with AI enhancements
    """
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.ai_imputer = AIImputer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.scaler = StandardScaler()
        
        # Component weights for quality scoring
        self.component_weights = {
            'consumption': 0.4,
            'investment': 0.2,
            'government_spending': 0.2,
            'net_exports': 0.2
        }
        
    async def calculate_gdp(
        self,
        data: ExpenditureData,
        country_code: str,
        period: str,
        apply_ai_corrections: bool = True
    ) -> GDPCalculationResult:
        """
        Calculate GDP using expenditure approach with AI enhancements
        
        Args:
            data: Expenditure components data
            country_code: ISO country code
            period: Time period (YYYY-Q1, YYYY-MM, YYYY)
            apply_ai_corrections: Whether to apply AI-based corrections
            
        Returns:
            GDPCalculationResult with computed GDP and metadata
        """
        logger.info(f"Calculating GDP for {country_code} - {period}")
        
        try:
            # 1. Data validation and preprocessing
            validated_data = await self._validate_and_preprocess(data, country_code)
            
            # 2. AI-assisted missing data imputation
            if apply_ai_corrections:
                imputed_data = await self._ai_impute_missing_values(
                    validated_data, country_code, period
                )
            else:
                imputed_data = validated_data
            
            # 3. Anomaly detection and correction
            anomaly_flags = await self._detect_anomalies(imputed_data, country_code)
            
            # 4. Calculate GDP components
            components = self._calculate_components(imputed_data)
            
            # 5. Compute final GDP
            gdp_value = self._compute_gdp_value(components)
            
            # 6. Uncertainty quantification
            uncertainty_bounds = await self._quantify_uncertainty(
                components, country_code, period
            )
            
            # 7. Quality assessment
            quality_score = self._assess_data_quality(imputed_data, anomaly_flags)
            
            # 8. Generate metadata
            metadata = self._generate_metadata(
                data, imputed_data, anomaly_flags, quality_score
            )
            
            result = GDPCalculationResult(
                gdp_value=gdp_value,
                components=components,
                country_code=country_code,
                period=period,
                method="expenditure",
                confidence_interval=uncertainty_bounds,
                quality_score=quality_score,
                anomaly_flags=anomaly_flags,
                metadata=metadata,
                calculation_timestamp=datetime.utcnow()
            )
            
            logger.info(f"GDP calculation completed: {gdp_value:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in GDP calculation: {str(e)}", exc_info=True)
            raise
    
    async def _validate_and_preprocess(
        self, 
        data: ExpenditureData, 
        country_code: str
    ) -> ExpenditureData:
        """Validate and preprocess input data"""
        
        # Basic validation
        if not country_code or len(country_code) != 3:
            raise ValueError("Invalid country code")
        
        # Check for minimum required data
        required_components = ['consumption', 'investment', 'government_spending']
        missing_components = []
        
        for component in required_components:
            if getattr(data, component) is None:
                missing_components.append(component)
        
        if len(missing_components) > 2:
            raise ValueError(f"Too many missing components: {missing_components}")
        
        # Data type validation and conversion
        validated_data = ExpenditureData(
            consumption=self._safe_float_conversion(data.consumption),
            investment=self._safe_float_conversion(data.investment),
            government_spending=self._safe_float_conversion(data.government_spending),
            exports=self._safe_float_conversion(data.exports),
            imports=self._safe_float_conversion(data.imports),
            timestamp=data.timestamp or datetime.utcnow(),
            source=data.source,
            confidence_score=data.confidence_score
        )
        
        return validated_data
    
    async def _ai_impute_missing_values(
        self,
        data: ExpenditureData,
        country_code: str,
        period: str
    ) -> ExpenditureData:
        """Use AI to impute missing values based on correlations and historical patterns"""
        
        # Prepare feature vector for imputation
        features = {
            'consumption': data.consumption,
            'investment': data.investment,
            'government_spending': data.government_spending,
            'exports': data.exports,
            'imports': data.imports
        }
        
        # Use AI imputer to fill missing values
        imputed_features = await self.ai_imputer.impute_missing_expenditure(
            features, country_code, period
        )
        
        # Create imputed data object
        imputed_data = ExpenditureData(
            consumption=imputed_features.get('consumption', data.consumption),
            investment=imputed_features.get('investment', data.investment),
            government_spending=imputed_features.get('government_spending', data.government_spending),
            exports=imputed_features.get('exports', data.exports),
            imports=imputed_features.get('imports', data.imports),
            timestamp=data.timestamp,
            source=data.source + " (AI-enhanced)",
            confidence_score=data.confidence_score * 0.9  # Slight penalty for imputation
        )
        
        return imputed_data
    
    async def _detect_anomalies(
        self,
        data: ExpenditureData,
        country_code: str
    ) -> Dict[str, bool]:
        """Detect anomalies in expenditure components"""
        
        # Prepare data for anomaly detection
        features_array = np.array([
            data.consumption or 0,
            data.investment or 0,
            data.government_spending or 0,
            data.exports or 0,
            data.imports or 0
        ]).reshape(1, -1)
        
        # Detect anomalies using multiple methods
        anomaly_results = await self.anomaly_detector.detect_expenditure_anomalies(
            features_array, country_code
        )
        
        return anomaly_results
    
    def _calculate_components(self, data: ExpenditureData) -> ExpenditureComponents:
        """Calculate GDP expenditure components"""
        
        # Ensure all components are available
        consumption = data.consumption or 0
        investment = data.investment or 0
        government_spending = data.government_spending or 0
        exports = data.exports or 0
        imports = data.imports or 0
        
        # Calculate net exports
        net_exports = exports - imports
        
        return ExpenditureComponents(
            consumption=consumption,
            investment=investment,
            government_spending=government_spending,
            exports=exports,
            imports=imports,
            net_exports=net_exports
        )
    
    def _compute_gdp_value(self, components: ExpenditureComponents) -> float:
        """Compute final GDP value from components"""
        
        gdp = (
            components.consumption +
            components.investment +
            components.government_spending +
            components.net_exports
        )
        
        return round(gdp, 2)
    
    async def _quantify_uncertainty(
        self,
        components: ExpenditureComponents,
        country_code: str,
        period: str
    ) -> Tuple[float, float]:
        """Quantify uncertainty bounds for GDP estimate"""
        
        # Use Bayesian inference for uncertainty quantification
        uncertainty_bounds = await self.uncertainty_quantifier.calculate_expenditure_uncertainty(
            components, country_code, period
        )
        
        return uncertainty_bounds
    
    def _assess_data_quality(
        self,
        data: ExpenditureData,
        anomaly_flags: Dict[str, bool]
    ) -> float:
        """Assess overall data quality score"""
        
        quality_factors = []
        
        # Completeness score
        total_components = 5
        non_null_components = sum([
            1 for value in [
                data.consumption, data.investment, data.government_spending,
                data.exports, data.imports
            ]
            if value is not None
        ])
        completeness_score = non_null_components / total_components
        quality_factors.append(completeness_score * 0.4)
        
        # Anomaly score (inverse of anomaly count)
        anomaly_count = sum(anomaly_flags.values())
        anomaly_score = max(0, 1 - (anomaly_count / len(anomaly_flags)))
        quality_factors.append(anomaly_score * 0.3)
        
        # Source confidence score
        quality_factors.append(data.confidence_score * 0.3)
        
        return round(sum(quality_factors), 3)
    
    def _generate_metadata(
        self,
        original_data: ExpenditureData,
        processed_data: ExpenditureData,
        anomaly_flags: Dict[str, bool],
        quality_score: float
    ) -> Dict:
        """Generate calculation metadata"""
        
        return {
            "method": "expenditure_approach",
            "ai_enhanced": True,
            "data_source": original_data.source,
            "quality_score": quality_score,
            "anomalies_detected": sum(anomaly_flags.values()),
            "anomaly_details": anomaly_flags,
            "imputation_applied": any([
                processed_data.consumption != original_data.consumption,
                processed_data.investment != original_data.investment,
                processed_data.government_spending != original_data.government_spending,
                processed_data.exports != original_data.exports,
                processed_data.imports != original_data.imports
            ]),
            "calculation_version": "1.0.0",
            "processor": "ExpenditureCalculator"
        }
    
    def _safe_float_conversion(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


# Factory function for calculator
def create_expenditure_calculator() -> ExpenditureCalculator:
    """Create and return configured expenditure calculator"""
    return ExpenditureCalculator()
