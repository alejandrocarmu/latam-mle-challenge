import logging
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from rich.logging import RichHandler
from contextlib import asynccontextmanager

from .model import DelayModel
from .utils import format_features_as_key_value  # Import the helper function

# Configure the logger with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

logger = logging.getLogger("DelayModelAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    Handles startup and shutdown events.
    """
    # Instantiate and load the model
    app.state.delay_model = DelayModel()
    
    logger.info("Attempting to load the model.")
    try:
        app.state.delay_model.load_model()
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise HTTPException(status_code=500, detail="Model files not found.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load the model.")
    logger.info("Model loaded successfully.")
    
    yield  # Application is now running
    
    # Shutdown logic
    logger.info("Shutting down the Flight Delay Prediction API.")

# Instantiate the FastAPI application with the lifespan handler
app = FastAPI(
    title="Flight Delay Prediction API",
    lifespan=lifespan
)

# Define Pydantic models for request and response
class Flight(BaseModel):
    OPERA: str = Field(..., json_schema_extra={"example": "Aerolineas Argentinas"})
    TIPOVUELO: str = Field(..., json_schema_extra={"example": "N"})  # 'I' for International, 'N' for National
    MES: int = Field(..., ge=1, le=12, json_schema_extra={"example": 3})  # Month number (1-12)

class PredictRequest(BaseModel):
    flights: List[Flight]

class PredictResponse(BaseModel):
    predict: List[int]

@app.get("/", status_code=200)
async def root():
    """
    Root endpoint providing a welcome message.
    """
    logger.info("Root endpoint called.")
    return {"message": "Welcome to the Flight Delay Prediction API!"}

@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint to verify if the API is running.
    """
    logger.info("Health check endpoint called.")
    return {"status": "OK"}

@app.get("/is_model_loaded")
async def is_model_loaded():
    """
    Endpoint to check if the model is loaded.
    """
    if app.state.delay_model and app.state.delay_model._model:
        return {"model_loaded": True}
    else:
        return {"model_loaded": False}

@app.post("/predict", response_model=PredictResponse, status_code=200)
async def post_predict(request: PredictRequest) -> PredictResponse:
    """
    Predict delays for a list of flights.
    
    Args:
        request (PredictRequest): The request body containing flight information.
    
    Returns:
        PredictResponse: The prediction results.
    """
    logger.info("Received a prediction request.")
    
    # Convert the request data to a DataFrame
    try:
        # Use .model_dump() instead of .dict() for Pydantic v2
        flights_data = [flight.model_dump() for flight in request.flights]
        data_df = pd.DataFrame(flights_data)
        
        # Log the received data in a key-value format
        data_str = data_df.to_string(index=False)
        logger.info(f"Received data:\n{data_str}")
    except Exception as ve:
        logger.error(f"Data conversion error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid input data.")

    # Simplified Preprocessing for API Use
    try:
        features = app.state.delay_model.api_preprocess(data=data_df)  # Use simplified method
        
        # Format the preprocessed features as key-value pairs
        features_str = format_features_as_key_value(features)
        logger.info(f"Preprocessed features:\n{features_str}")
    except KeyError as ke:
        logger.error(f"Preprocessing error: Missing column {ke}")
        raise HTTPException(status_code=400, detail=f"Missing feature columns: {', '.join(ke.args)}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing failed.")

    # Predict using the loaded model
    try:
        predictions = app.state.delay_model.predict(features)
        logger.info(f"Predictions: {predictions}")
    except KeyError as e:
        logger.error(f"Prediction error: Missing columns {e}")
        raise HTTPException(status_code=400, detail=f"Missing feature columns: {', '.join(e.args)}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictResponse(predict=predictions)
