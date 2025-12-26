"""
Forecast API Endpoints

Provides endpoints for:
- Data upload and validation
- SARIMA demand forecasting by category
- Hybrid Ensemble forecasting (SARIMA + Prophet + LSTM)
- Inventory recommendations
- Model retraining
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import io
import logging

from ..database import get_db
from ..ml.sarima_forecaster import CategoryForecaster, load_and_prepare_data, MODEL_DIR

# Try to import ensemble forecaster (may fail if dependencies not installed)
try:
    from ..ml.ensemble_forecaster import EnsembleForecaster, CategoryEnsembleForecaster
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("Ensemble forecaster not available. Install prophet and tensorflow.")

router = APIRouter()
logger = logging.getLogger(__name__)

# Global forecaster instances
_forecaster: Optional[CategoryForecaster] = None
_ensemble_forecaster: Optional['CategoryEnsembleForecaster'] = None


# ==================== Response Models ====================

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class DailyForecast(BaseModel):
    date: str
    forecast: float
    lower_ci: float
    upper_ci: float


class ForecastResponse(BaseModel):
    category: str
    forecast_90_day: float
    confidence_interval: ConfidenceInterval
    daily_forecast: Optional[List[DailyForecast]] = None
    reorder_point: float
    safety_stock: float
    stockout_risk: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    records_processed: int
    categories_found: List[str]
    date_range: dict


class RetrainResponse(BaseModel):
    success: bool
    message: str
    models_trained: int
    training_results: List[dict]


class InventoryRecommendation(BaseModel):
    category: str
    forecast_90_day: float
    confidence_interval: ConfidenceInterval
    daily_average: float
    reorder_point: float
    safety_stock: float
    stockout_risk: str
    recommended_order_quantity: float


# ==================== Helper Functions ====================

def get_forecaster() -> CategoryForecaster:
    """Get or create forecaster instance."""
    global _forecaster
    if _forecaster is None:
        _forecaster = CategoryForecaster()
    return _forecaster


def validate_csv_columns(df: pd.DataFrame) -> tuple:
    """
    Validate required columns exist in uploaded CSV.

    Returns:
        Tuple of (is_valid, missing_columns or None)
    """
    required = {'order_date', 'quantity', 'category'}
    alt_names = {
        'order_date': ['orderdate', 'date', 'order_date'],
        'quantity': ['quantity', 'qty', 'units'],
        'category': ['category', 'product_category', 'cat']
    }

    df_cols = {c.lower() for c in df.columns}

    missing = []
    for req in required:
        found = False
        for alt in alt_names[req]:
            if alt in df_cols:
                found = True
                break
        if not found:
            missing.append(req)

    return (len(missing) == 0, missing if missing else None)


# ==================== Endpoints ====================

@router.post("/upload-data", response_model=UploadResponse)
async def upload_sales_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload sales data CSV for forecasting.

    Required columns:
    - order_date (or date): Order date
    - quantity (or qty): Quantity sold
    - category: Product category

    Optional columns:
    - product_id: Product identifier
    - order_status: Order status (for filtering cancelled orders)
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file"
        )

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate columns
        is_valid, missing = validate_csv_columns(df)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}. Required: order_date, quantity, category"
            )

        # Prepare data
        df = load_and_prepare_data(io.StringIO(contents.decode('utf-8')))

        # Initialize forecaster with data
        global _forecaster
        _forecaster = CategoryForecaster(df)

        # Get categories and date range
        cat_col = 'Category' if 'Category' in df.columns else 'category'
        date_col = 'OrderDate' if 'OrderDate' in df.columns else 'order_date'

        categories = df[cat_col].unique().tolist()

        # Trigger model training in background
        if background_tasks:
            background_tasks.add_task(_forecaster.train_all_categories)

        return UploadResponse(
            success=True,
            message="Data uploaded successfully. Models will be trained in background.",
            records_processed=len(df),
            categories_found=categories,
            date_range={
                'start': df[date_col].min().strftime('%Y-%m-%d'),
                'end': df[date_col].max().strftime('%Y-%m-%d')
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/forecast/{category}", response_model=ForecastResponse)
async def get_category_forecast(
    category: str,
    horizon: int = Query(90, ge=7, le=365, description="Forecast horizon in days"),
    include_daily: bool = Query(False, description="Include daily forecast details")
):
    """
    Get SARIMA forecast for a specific category.

    Returns:
    - 90-day total forecast
    - Confidence intervals
    - Reorder point and safety stock recommendations
    - Stockout risk assessment
    """
    forecaster = get_forecaster()

    try:
        # Generate forecast
        result = forecaster.forecast_category(category, horizons=[30, 60, horizon])

        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])

        # Get the requested horizon forecast
        horizon_key = f'{horizon}_day'
        forecast_data = result['forecasts'].get(horizon_key, result['forecasts'].get('90_day'))

        # Get recommendations
        recs = forecaster.get_inventory_recommendations()
        cat_rec = next((r for r in recs if r['category'] == category), None)

        if cat_rec and 'error' not in cat_rec:
            reorder_point = cat_rec['reorder_point']
            safety_stock = cat_rec['safety_stock']
            stockout_risk = cat_rec['stockout_risk']
        else:
            # Calculate basic values
            total = forecast_data['total_forecast']
            reorder_point = round(total / horizon * 14, 0)
            safety_stock = round(total * 0.1, 0)
            stockout_risk = 'medium'

        response = ForecastResponse(
            category=category,
            forecast_90_day=forecast_data['total_forecast'],
            confidence_interval=ConfidenceInterval(
                lower=forecast_data['lower_ci_total'],
                upper=forecast_data['upper_ci_total']
            ),
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            stockout_risk=stockout_risk
        )

        if include_daily and 'daily_forecast' in forecast_data:
            response.daily_forecast = [
                DailyForecast(**d) for d in forecast_data['daily_forecast']
            ]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forecasting {category}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )


@router.get("/forecasts/all")
async def get_all_forecasts(
    horizon: int = Query(90, ge=7, le=365)
):
    """
    Get forecasts for all categories.

    Returns summary forecasts for each category including:
    - Total forecast for the horizon
    - Confidence intervals
    - Risk assessment
    """
    forecaster = get_forecaster()

    try:
        all_forecasts = forecaster.forecast_all_categories(horizons=[30, 60, horizon])

        results = []
        total_forecast = 0

        for category, data in all_forecasts.items():
            if 'error' in data:
                results.append({
                    'category': category,
                    'error': data['error']
                })
                continue

            horizon_key = f'{horizon}_day'
            forecast_data = data['forecasts'].get(horizon_key, {})

            cat_forecast = forecast_data.get('total_forecast', 0)
            total_forecast += cat_forecast

            results.append({
                'category': category,
                f'forecast_{horizon}_day': cat_forecast,
                'confidence_interval': {
                    'lower': forecast_data.get('lower_ci_total', 0),
                    'upper': forecast_data.get('upper_ci_total', 0)
                }
            })

        return {
            'generated_at': datetime.now().isoformat(),
            'horizon_days': horizon,
            'total_forecast_all_categories': round(total_forecast, 0),
            'categories': results
        }

    except Exception as e:
        logger.error(f"Error getting all forecasts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecasts: {str(e)}"
        )


@router.get("/inventory-recommendations", response_model=List[InventoryRecommendation])
async def get_inventory_recommendations():
    """
    Get inventory management recommendations based on forecasts.

    Returns for each category:
    - Reorder points
    - Safety stock levels
    - Stockout risk assessment
    - Recommended order quantities
    """
    forecaster = get_forecaster()

    try:
        recommendations = forecaster.get_inventory_recommendations()

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No trained models found. Please upload data first."
            )

        # Filter out errors and format response
        valid_recs = []
        for rec in recommendations:
            if 'error' not in rec:
                valid_recs.append(InventoryRecommendation(
                    category=rec['category'],
                    forecast_90_day=rec['forecast_90_day'],
                    confidence_interval=ConfidenceInterval(
                        lower=rec['confidence_interval']['lower'],
                        upper=rec['confidence_interval']['upper']
                    ),
                    daily_average=rec['daily_average'],
                    reorder_point=rec['reorder_point'],
                    safety_stock=rec['safety_stock'],
                    stockout_risk=rec['stockout_risk'],
                    recommended_order_quantity=rec['recommended_order_quantity']
                ))

        return valid_recs

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/retrain-models", response_model=RetrainResponse)
async def retrain_models(
    background_tasks: BackgroundTasks = None,
    force: bool = Query(False, description="Force retrain even if models exist")
):
    """
    Trigger model retraining with current data.

    Models are trained in the background and saved to disk.
    Use force=True to overwrite existing models.
    """
    forecaster = get_forecaster()

    if forecaster.df is None:
        raise HTTPException(
            status_code=400,
            detail="No data available. Please upload data first using /upload-data"
        )

    try:
        # Train all models
        results = forecaster.train_all_categories()

        successful = [r for r in results if 'error' not in r]

        return RetrainResponse(
            success=True,
            message=f"Successfully trained {len(successful)} models",
            models_trained=len(successful),
            training_results=results
        )

    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retraining models: {str(e)}"
        )


@router.get("/model-status")
async def get_model_status():
    """
    Get status of trained models.

    Returns list of available models and their metadata.
    """
    try:
        model_files = list(MODEL_DIR.glob('sarima_*.pkl'))

        models = []
        for path in model_files:
            category = path.stem.replace('sarima_', '').replace('_', ' ').replace('and', '&')
            stat = path.stat()
            models.append({
                'category': category,
                'model_file': path.name,
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'size_bytes': stat.st_size
            })

        return {
            'models_available': len(models),
            'model_directory': str(MODEL_DIR),
            'models': models
        }

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking models: {str(e)}"
        )


# ==================== Ensemble Forecasting Endpoints ====================

class EnsembleWeights(BaseModel):
    sarima: float
    prophet: float
    lstm: float


class EnsembleForecastResponse(BaseModel):
    category: str
    forecast: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    weights: EnsembleWeights
    horizon_weeks: int
    model_type: str = "ensemble"


class EnsembleTrainResponse(BaseModel):
    success: bool
    message: str
    categories_trained: int
    results: List[Dict[str, Any]]


@router.get("/ensemble/status")
async def get_ensemble_status():
    """
    Check if ensemble forecasting is available.

    Returns availability status and installed dependencies.
    """
    return {
        "ensemble_available": ENSEMBLE_AVAILABLE,
        "models": ["SARIMA", "Prophet", "LSTM"] if ENSEMBLE_AVAILABLE else [],
        "message": "Ensemble forecasting ready" if ENSEMBLE_AVAILABLE else "Install prophet and tensorflow for ensemble forecasting"
    }


@router.post("/ensemble/train", response_model=EnsembleTrainResponse)
async def train_ensemble_models(
    file: UploadFile = File(...),
    validation_weeks: int = Query(8, ge=4, le=16, description="Weeks for weight optimization")
):
    """
    Train hybrid ensemble models (SARIMA + Prophet + LSTM) for all categories.

    The ensemble uses intelligent weight optimization:
    1. Each model is validated on a holdout period
    2. Weights are assigned inversely proportional to validation SMAPE
    3. Better performing models get higher weights

    Target: SMAPE < 20%
    """
    if not ENSEMBLE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Ensemble forecasting not available. Install: pip install prophet tensorflow"
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate columns
        is_valid, missing = validate_csv_columns(df)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        # Prepare data
        df = load_and_prepare_data(io.StringIO(contents.decode('utf-8')))

        # Initialize and train ensemble
        global _ensemble_forecaster
        _ensemble_forecaster = CategoryEnsembleForecaster(validation_size=validation_weeks)
        _ensemble_forecaster.fit_all_categories(
            df,
            date_col='OrderDate' if 'OrderDate' in df.columns else 'order_date',
            category_col='Category' if 'Category' in df.columns else 'category',
            value_col='Quantity' if 'Quantity' in df.columns else 'quantity'
        )

        # Get summary
        summary = _ensemble_forecaster.get_summary()
        successful = len(summary[summary['status'] == 'success'])

        results = []
        for _, row in summary.iterrows():
            if row['status'] == 'success':
                results.append({
                    'category': row['category'],
                    'weights': {
                        'sarima': round(row['sarima_weight'], 3),
                        'prophet': round(row['prophet_weight'], 3),
                        'lstm': round(row['lstm_weight'], 3)
                    },
                    'data_points': row['data_points'],
                    'status': 'success'
                })
            else:
                results.append({
                    'category': row['category'],
                    'status': 'failed'
                })

        return EnsembleTrainResponse(
            success=True,
            message=f"Trained ensemble models for {successful} categories",
            categories_trained=successful,
            results=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training ensemble: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error training ensemble models: {str(e)}"
        )


@router.get("/ensemble/forecast/{category}", response_model=EnsembleForecastResponse)
async def get_ensemble_forecast(
    category: str,
    weeks: int = Query(4, ge=1, le=52, description="Forecast horizon in weeks")
):
    """
    Get hybrid ensemble forecast for a specific category.

    Combines SARIMA, Prophet, and LSTM predictions using optimized weights.
    Returns weekly forecasts with confidence intervals.
    """
    if not ENSEMBLE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Ensemble forecasting not available"
        )

    global _ensemble_forecaster
    if _ensemble_forecaster is None:
        raise HTTPException(
            status_code=400,
            detail="No ensemble models trained. Use /ensemble/train first."
        )

    try:
        result = _ensemble_forecaster.predict_category(category, steps=weeks)

        return EnsembleForecastResponse(
            category=result['category'],
            forecast=result['forecast'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            weights=EnsembleWeights(**result['weights']),
            horizon_weeks=result['horizon_weeks']
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating ensemble forecast: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )


@router.get("/ensemble/forecasts/all")
async def get_all_ensemble_forecasts(
    weeks: int = Query(4, ge=1, le=52)
):
    """
    Get ensemble forecasts for all trained categories.

    Returns forecasts with model weights for each category.
    """
    if not ENSEMBLE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Ensemble forecasting not available"
        )

    global _ensemble_forecaster
    if _ensemble_forecaster is None:
        raise HTTPException(
            status_code=400,
            detail="No ensemble models trained. Use /ensemble/train first."
        )

    try:
        results = []
        for category in _ensemble_forecaster.forecasters.keys():
            try:
                result = _ensemble_forecaster.predict_category(category, steps=weeks)
                results.append({
                    'category': result['category'],
                    'forecast_total': sum(result['forecast']),
                    'weekly_forecast': result['forecast'],
                    'weights': result['weights']
                })
            except Exception as e:
                results.append({
                    'category': category,
                    'error': str(e)
                })

        return {
            'generated_at': datetime.now().isoformat(),
            'horizon_weeks': weeks,
            'categories': results
        }

    except Exception as e:
        logger.error(f"Error generating ensemble forecasts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecasts: {str(e)}"
        )


@router.get("/ensemble/weights")
async def get_ensemble_weights():
    """
    Get optimized weights for all trained ensemble models.

    Shows how much each model (SARIMA, Prophet, LSTM) contributes
    to the final forecast for each category.
    """
    global _ensemble_forecaster
    if _ensemble_forecaster is None:
        raise HTTPException(
            status_code=400,
            detail="No ensemble models trained"
        )

    summary = _ensemble_forecaster.get_summary()

    return {
        'categories': summary.to_dict(orient='records'),
        'average_weights': {
            'sarima': summary['sarima_weight'].mean(),
            'prophet': summary['prophet_weight'].mean(),
            'lstm': summary['lstm_weight'].mean()
        }
    }


@router.get("/ensemble/compare/{category}")
async def compare_models(
    category: str,
    weeks: int = Query(4, ge=1, le=12)
):
    """
    Compare SARIMA-only forecast vs Ensemble forecast for a category.

    Useful for evaluating ensemble improvement over single models.
    """
    if not ENSEMBLE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Ensemble forecasting not available"
        )

    results = {
        'category': category,
        'horizon_weeks': weeks
    }

    # Get SARIMA forecast
    try:
        forecaster = get_forecaster()
        sarima_result = forecaster.forecast_category(category, horizons=[weeks * 7])
        horizon_key = f'{weeks * 7}_day'
        if 'forecasts' in sarima_result and horizon_key in sarima_result['forecasts']:
            results['sarima'] = {
                'total_forecast': sarima_result['forecasts'][horizon_key]['total_forecast'],
                'model': 'SARIMA only'
            }
    except Exception as e:
        results['sarima'] = {'error': str(e)}

    # Get Ensemble forecast
    global _ensemble_forecaster
    if _ensemble_forecaster:
        try:
            ensemble_result = _ensemble_forecaster.predict_category(category, steps=weeks)
            results['ensemble'] = {
                'total_forecast': sum(ensemble_result['forecast']),
                'weights': ensemble_result['weights'],
                'model': 'SARIMA + Prophet + LSTM'
            }
        except Exception as e:
            results['ensemble'] = {'error': str(e)}
    else:
        results['ensemble'] = {'error': 'Ensemble not trained'}

    return results
