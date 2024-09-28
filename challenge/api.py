import os
import logging
from typing import List

import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from rich.logging import RichHandler

from .model import DelayModel

# Configure the logger with RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

logger = logging.getLogger("DelayModelAPI")

app = FastAPI(title="Flight Delay Prediction API")


# Define Pydantic models for request and response
class Flight(BaseModel):
    OPERA: str = Field(..., example="Aerolineas Argentinas")
    TIPOVUELO: str = Field(..., example="N")  # 'I' for International, 'N' for National
    MES: int = Field(..., ge=1, le=12, example=3)  # Month number (1-12)


class PredictRequest(BaseModel):
    flights: List[Flight]


class PredictResponse(BaseModel):
    predict: List[int]


@app.on_event("startup")
def load_model_event():
    """
    Load the trained model and feature names from the models directory during startup.
    """
    global delay_model

    logger.info("Loading the trained model and feature names.")
    delay_model = DelayModel()

    try:
        delay_model.load_model()
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise e

    logger.info("Model loaded successfully.")


@app.on_event("shutdown")
def shutdown_event():
    """
    Perform any necessary cleanup during shutdown.
    """
    logger.info("Shutting down the Flight Delay Prediction API.")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """
    Health check endpoint to verify if the API is running.
    """
    logger.info("Health check endpoint called.")
    return {"status": "OK"}


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
        flights_data = [flight.dict() for flight in request.flights]
        data_df = pd.DataFrame(flights_data)
        logger.info(f"Received data: {data_df}")
    except Exception as ve:
        logger.error(f"Data conversion error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid input data.")

    # Preprocess the data
    try:
        features = delay_model.preprocess(data=data_df)
        logger.info(f"Preprocessed features: {features}")
    except KeyError as ke:
        logger.error(f"Preprocessing error: Missing column {ke}")
        raise HTTPException(status_code=400, detail=f"Missing column: {ke}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail="Preprocessing failed.")

    # Predict using the loaded model
    try:
        predictions = delay_model.predict(features)
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictResponse(predict=predictions)