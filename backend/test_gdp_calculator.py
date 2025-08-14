"""
Tests for GDP Calculator functionality
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from core.gdp_calculation.expenditure_calculator import ExpenditureCalculator, ExpenditureData
from core.models.gdp_models import GDPCalculationResult
from services.gdp_service import GDPService


class TestExpenditureCalculator:
    """Test cases for Expenditure Calculator"""
    
    @pytest.fixture
    def calculator(self):
        return ExpenditureCalculator()
    
    @pytest.fixture
    def sample_data(self):
        return ExpenditureData(
            consumption=18500.0,
            investment=4800.0,
            government_spending=4200.0,
            exports=2800.0,
            imports=3300.0,
            timestamp=datetime.utcnow(),
            source="Test Data",
            confidence_score=0.95
        )
    
    def test_basic_gdp_calculation(self, calculator, sample_data):
        """Test basic GDP calculation without AI corrections"""
        
        # Mock AI components to avoid external dependencies
        with patch.object(calculator, 'anomaly_detector'), \
             patch.object(calculator, 'ai_imputer'), \
             patch.object(calculator, 'uncertainty_quantifier'):
            
            result = calculator.calculate_gdp(
                data=sample_data,
                country_code="USA",
                period="2024-Q1",
                apply_ai_corrections=False
            )
        
        # Expected GDP = C + I + G + (X - M) = 18500 + 4800 + 4200 + (2800 - 3300) = 26500
        assert result.gdp_value == 26500.0
        assert result.country_code == "USA"
        assert result.period == "2024-Q1"
        assert result.method == "expenditure"
    
    def test_gdp_components_calculation(self, calculator, sample_data):
        """Test component calculation accuracy"""
        
        components = calculator._calculate_components(sample_data)
        
        assert components.consumption == 18500.0
        assert components.investment == 4800.0
        assert components.government_spending == 4200.0
        assert components.exports == 2800.0
        assert components.imports == 3300.0
        assert components.net_exports == -500.0  # 2800 - 3300
    
    def test_data_validation(self, calculator):
        """Test data validation functionality"""
        
        # Test valid data
        valid_data = ExpenditureData(
            consumption=1000.0,
            investment=500.0,
            government_spending=300.0,
            exports=200.0,
            imports=150.0
        )
        
        validated = calculator._validate_and_preprocess(valid_data, "USA")
        assert validated.consumption == 1000.0
        
        # Test invalid country code
        with pytest.raises(ValueError, match="Invalid country code"):
            calculator._validate_and_preprocess(valid_data, "INVALID")
    
    def test_missing_data_handling(self, calculator):
        """Test handling of missing data components"""
        
        # Test data with missing components
        incomplete_data = ExpenditureData(
            consumption=1000.0,
            investment=None,  # Missing
            government_spending=300.0,
            exports=None,     # Missing
            imports=150.0
        )
        
        # Should not raise error for some missing components
        validated = calculator._validate_and_preprocess(incomplete_data, "USA")
        assert validated.consumption == 1000.0
        assert validated.investment is None
        
        # Test too many missing components
        too_incomplete_data = ExpenditureData(
            consumption=None,
            investment=None,
            government_spending=None,
            exports=200.0,
            imports=150.0
        )
        
        with pytest.raises(ValueError, match="Too many missing components"):
            calculator._validate_and_preprocess(too_incomplete_data, "USA")
    
    def test_quality_assessment(self, calculator, sample_data):
        """Test data quality scoring"""
        
        anomaly_flags = {
            'consumption': False,
            'investment': True,  # One anomaly
            'government_spending': False,
            'exports': False,
            'imports': False
        }
        
        quality_score = calculator._assess_data_quality(sample_data, anomaly_flags)
        
        # Quality score should be reduced due to anomaly
        assert 0.0 <= quality_score <= 1.0
        assert quality_score < 1.0  # Should be less than perfect due to anomaly
    
    @pytest.mark.asyncio
    async def test_ai_corrections(self, calculator, sample_data):
        """Test AI corrections functionality"""
        
        # Mock AI imputer to return modified data
        mock_imputer = Mock()
        mock_imputer.impute_missing_expenditure.return_value = {
            'consumption': 18600.0,  # Slightly modified
            'investment': 4800.0,
            'government_spending': 4200.0,
            'exports': 2800.0,
            'imports': 3300.0
        }
        calculator.ai_imputer = mock_imputer
        
        # Mock other AI components
        calculator.anomaly_detector = Mock()
        calculator.anomaly_detector.detect_expenditure_anomalies.return_value = {
            'consumption': False,
            'investment': False,
            'government_spending': False,
            'exports': False,
            'imports': False
        }
        
        calculator.uncertainty_quantifier = Mock()
        calculator.uncertainty_quantifier.calculate_expenditure_uncertainty.return_value = (26400.0, 26600.0)
        
        result = await calculator.calculate_gdp(
            data=sample_data,
            country_code="USA",
            period="2024-Q1",
            apply_ai_corrections=True
        )
        
        # Should use AI-corrected consumption value
        assert result.gdp_value == 26600.0  # 18600 + 4800 + 4200 - 500
        assert result.confidence_interval == (26400.0, 26600.0)


class TestGDPService:
    """Test cases for GDP Service"""
    
    @pytest.fixture
    def gdp_service(self):
        return GDPService()
    
    @pytest.fixture
    def mock_db_session(self):
        return Mock()
    
    def test_service_initialization(self, gdp_service):
        """Test GDP service initialization"""
        assert gdp_service is not None
        assert hasattr(gdp_service, 'expenditure_calculator')
        assert hasattr(gdp_service, 'income_calculator')
        assert hasattr(gdp_service, 'output_calculator')
    
    @pytest.mark.asyncio
    async def test_historical_data_retrieval(self, gdp_service, mock_db_session):
        """Test historical GDP data retrieval"""
        
        # Mock database query results
        mock_records = [
            Mock(
                country_code="USA",
                period="2024-Q1",
                gdp_value=26500.0,
                method="expenditure",
                components={"consumption": 18500.0},
                created_at=datetime.utcnow()
            ),
            Mock(
                country_code="USA",
                period="2023-Q4",
                gdp_value=26300.0,
                method="expenditure",
                components={"consumption": 18300.0},
                created_at=datetime.utcnow()
            )
        ]
        
        with patch.object(gdp_service, '_query_historical_records', return_value=mock_records):
            result = await gdp_service.get_historical_data(
                country_code="USA",
                start_date="2023-01-01",
                end_date="2024-12-31",
                db=mock_db_session
            )
        
        assert len(result['records']) == 2
        assert result['records'][0]['gdp_value'] == 26500.0
        assert result['records'][1]['gdp_value'] == 26300.0
    
    @pytest.mark.asyncio
    async def test_country_comparison(self, gdp_service, mock_db_session):
        """Test country GDP comparison"""
        
        mock_data = {
            "USA": [{"period": "2024-Q1", "gdp_value": 26500.0}],
            "CHN": [{"period": "2024-Q1", "gdp_value": 17700.0}],
            "JPN": [{"period": "2024-Q1", "gdp_value": 4200.0}]
        }
        
        with patch.object(gdp_service, '_get_country_data', side_effect=lambda country, **kwargs: mock_data[country]):
            result = await gdp_service.compare_countries(
                country_codes=["USA", "CHN", "JPN"],
                period="2024-Q1",
                db=mock_db_session
            )
        
        assert len(result['countries']) == 3
        assert result['countries'][0]['country_code'] == "USA"
        assert result['countries'][0]['gdp_value'] == 26500.0
    
    @pytest.mark.asyncio
    async def test_data_validation(self, gdp_service):
        """Test input data validation"""
        
        valid_data = {
            'consumption': 18500.0,
            'investment': 4800.0,
            'government_spending': 4200.0,
            'exports': 2800.0,
            'imports': 3300.0
        }
        
        result = await gdp_service.validate_input_data(
            data=valid_data,
            country_code="USA",
            method="expenditure"
        )
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # Test invalid data
        invalid_data = {
            'consumption': -1000.0,  # Negative value
            'investment': 'invalid',  # Wrong type
            'government_spending': 4200.0
        }
        
        result = await gdp_service.validate_input_data(
            data=invalid_data,
            country_code="USA",
            method="expenditure"
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0


@pytest.mark.integration
class TestGDPIntegration:
    """Integration tests for GDP calculation system"""
    
    @pytest.fixture
    def test_database(self):
        """Setup test database"""
        # This would setup a test database instance
        # For example purposes, we'll use a mock
        return Mock()
    
    @pytest.mark.asyncio
    async def test_end_to_end_calculation(self, test_database):
        """Test complete GDP calculation flow"""
        
        # Setup test data
        input_data = ExpenditureData(
            consumption=18500.0,
            investment=4800.0,
            government_spending=4200.0,
            exports=2800.0,
            imports=3300.0,
            timestamp=datetime.utcnow(),
            source="Integration Test",
            confidence_score=0.95
        )
        
        # Initialize services
        calculator = ExpenditureCalculator()
        gdp_service = GDPService()
        
        # Mock AI components for integration test
        with patch.object(calculator, 'anomaly_detector'), \
             patch.object(calculator, 'ai_imputer'), \
             patch.object(calculator, 'uncertainty_quantifier'), \
             patch.object(gdp_service, 'store_calculation_result'):
            
            # Calculate GDP
            result = await calculator.calculate_gdp(
                data=input_data,
                country_code="USA",
                period="2024-Q1",
                apply_ai_corrections=True
            )
            
            # Store result
            await gdp_service.store_calculation_result(result, user_id="test_user")
        
        # Verify result
        assert result.gdp_value == 26500.0
        assert result.country_code == "USA"
        assert result.period == "2024-Q1"
        assert result.method == "expenditure"
        assert result.quality_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])