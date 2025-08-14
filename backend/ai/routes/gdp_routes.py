"""
GDP Calculation API Routes
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.database import get_db
from core.models.gdp_models import (
    GDPCalculationRequest, GDPCalculationResult, 
    APIResponse, PaginatedResponse
)
from core.gdp_calculation.expenditure_calculator import ExpenditureCalculator
from core.gdp_calculation.income_calculator import IncomeCalculator
from core.gdp_calculation.output_calculator import OutputCalculator
from services.gdp_service import GDPService
from services.auth_service import get_current_user
from core.monitoring.metrics import track_api_call

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize calculators
expenditure_calc = ExpenditureCalculator()
income_calc = IncomeCalculator()
output_calc = OutputCalculator()
gdp_service = GDPService()


@router.post("/calculate", response_model=APIResponse)
@track_api_call
async def calculate_gdp(
    request: GDPCalculationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Calculate GDP using specified method with AI enhancements
    
    Supports three calculation methods:
    - expenditure: C + I + G + (X - M)
    - income: Sum of all income components
    - output: Sum of sectoral gross value added
    """
    try:
        logger.info(f"GDP calculation requested for {request.country_code} - {request.period}")
        
        # Select appropriate calculator
        if request.method == "expenditure":
            calculator = expenditure_calc
        elif request.method == "income":
            calculator = income_calc
        elif request.method == "output":
            calculator = output_calc
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported calculation method: {request.method}"
            )
        
        # Perform calculation
        result = await calculator.calculate_gdp(
            data=request.data,
            country_code=request.country_code,
            period=request.period,
            apply_ai_corrections=request.apply_ai_corrections
        )
        
        # Store result in database
        background_tasks.add_task(
            gdp_service.store_calculation_result,
            result,
            current_user.id
        )
        
        # Log successful calculation
        logger.info(f"GDP calculation completed: {result.gdp_value}")
        
        return APIResponse(
            success=True,
            data=result.dict(),
            message="GDP calculation completed successfully"
        )
        
    except ValueError as e:
        logger.error(f"Validation error in GDP calculation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error in GDP calculation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/historical/{country_code}", response_model=APIResponse)
@track_api_call
async def get_historical_gdp(
    country_code: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    method: Optional[str] = Query(None, description="Calculation method filter"),
    include_components: bool = Query(False, description="Include component breakdown"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get historical GDP data for a country"""
    try:
        logger.info(f"Historical GDP requested for {country_code}")
        
        historical_data = await gdp_service.get_historical_data(
            country_code=country_code.upper(),
            start_date=start_date,
            end_date=end_date,
            method=method,
            include_components=include_components,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=historical_data,
            message=f"Historical GDP data retrieved for {country_code}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving historical GDP: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/compare", response_model=APIResponse)
@track_api_call
async def compare_countries_gdp(
    country_codes: List[str] = Query(..., description="List of country codes"),
    period: Optional[str] = Query(None, description="Specific period (YYYY-Q1)"),
    start_date: Optional[str] = Query(None, description="Start date for range"),
    end_date: Optional[str] = Query(None, description="End date for range"),
    normalize: bool = Query(False, description="Normalize by population"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Compare GDP across multiple countries"""
    try:
        logger.info(f"GDP comparison requested for {country_codes}")
        
        comparison_data = await gdp_service.compare_countries(
            country_codes=[code.upper() for code in country_codes],
            period=period,
            start_date=start_date,
            end_date=end_date,
            normalize=normalize,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=comparison_data,
            message=f"GDP comparison completed for {len(country_codes)} countries"
        )
        
    except Exception as e:
        logger.error(f"Error in GDP comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/components/{country_code}/{period}", response_model=APIResponse)
@track_api_call
async def get_gdp_components(
    country_code: str,
    period: str,
    method: str = Query("expenditure", description="Calculation method"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detailed GDP component breakdown"""
    try:
        logger.info(f"GDP components requested for {country_code} - {period}")
        
        components = await gdp_service.get_components_breakdown(
            country_code=country_code.upper(),
            period=period,
            method=method,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=components,
            message="GDP components retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving GDP components: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/validate", response_model=APIResponse)
@track_api_call
async def validate_gdp_data(
    request: Dict[str, Any],
    country_code: str,
    method: str,
    current_user = Depends(get_current_user)
):
    """Validate GDP data before calculation"""
    try:
        logger.info(f"GDP data validation requested for {country_code}")
        
        validation_result = await gdp_service.validate_input_data(
            data=request,
            country_code=country_code.upper(),
            method=method
        )
        
        return APIResponse(
            success=True,
            data=validation_result,
            message="Data validation completed"
        )
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/quality-report/{country_code}", response_model=APIResponse)
@track_api_call
async def get_data_quality_report(
    country_code: str,
    period_range: Optional[str] = Query(None, description="Period range (YYYY-YYYY)"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get data quality assessment report"""
    try:
        logger.info(f"Data quality report requested for {country_code}")
        
        quality_report = await gdp_service.generate_quality_report(
            country_code=country_code.upper(),
            period_range=period_range,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=quality_report,
            message="Data quality report generated"
        )
        
    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/bulk-calculate", response_model=APIResponse)
@track_api_call
async def bulk_calculate_gdp(
    requests: List[GDPCalculationRequest],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Bulk GDP calculation for multiple requests"""
    try:
        logger.info(f"Bulk GDP calculation requested for {len(requests)} items")
        
        # Process calculations in background
        background_tasks.add_task(
            gdp_service.process_bulk_calculations,
            requests,
            current_user.id,
            db
        )
        
        return APIResponse(
            success=True,
            data={"job_id": f"bulk_{datetime.utcnow().timestamp()}"},
            message=f"Bulk calculation job started for {len(requests)} items"
        )
        
    except Exception as e:
        logger.error(f"Error in bulk calculation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/trends/{country_code}", response_model=APIResponse)
@track_api_call
async def get_gdp_trends(
    country_code: str,
    lookback_periods: int = Query(12, description="Number of periods to analyze"),
    include_forecast: bool = Query(False, description="Include trend forecast"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Analyze GDP trends and patterns"""
    try:
        logger.info(f"GDP trends analysis requested for {country_code}")
        
        trends = await gdp_service.analyze_trends(
            country_code=country_code.upper(),
            lookback_periods=lookback_periods,
            include_forecast=include_forecast,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=trends,
            message="GDP trends analysis completed"
        )
        
    except Exception as e:
        logger.error(f"Error in trends analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/anomalies/{country_code}", response_model=APIResponse)
@track_api_call
async def detect_gdp_anomalies(
    country_code: str,
    sensitivity: float = Query(0.05, description="Anomaly detection sensitivity"),
    method: Optional[str] = Query(None, description="Detection method"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Detect anomalies in GDP data"""
    try:
        logger.info(f"GDP anomaly detection requested for {country_code}")
        
        anomalies = await gdp_service.detect_anomalies(
            country_code=country_code.upper(),
            sensitivity=sensitivity,
            method=method,
            db=db
        )
        
        return APIResponse(
            success=True,
            data=anomalies,
            message="GDP anomaly detection completed"
        )
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/metadata/standards", response_model=APIResponse)
@track_api_call
async def get_standards_compliance(
    standard: str = Query("SNA2008", description="Standard to check (SNA2008, BPM6)"),
    current_user = Depends(get_current_user)
):
    """Get information about standards compliance"""
    try:
        logger.info(f"Standards compliance info requested for {standard}")
        
        compliance_info = await gdp_service.get_standards_info(standard)
        
        return APIResponse(
            success=True,
            data=compliance_info,
            message=f"Standards compliance information for {standard}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving standards info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
