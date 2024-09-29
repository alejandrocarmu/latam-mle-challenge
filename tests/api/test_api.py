import unittest
from fastapi.testclient import TestClient
from challenge.api import app

class TestBatchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the TestClient and ensure the model is loaded before tests run.
        """
        # Use TestClient as a context manager to trigger lifespan events
        with TestClient(app) as client:
            cls.client = client
    
    def test_root_endpoint(self):
        """
        Test the root endpoint.
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to the Flight Delay Prediction API!"})
    
    def test_health_endpoint(self):
        """
        Test the health check endpoint.
        """
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "OK"})
    
    def test_is_model_loaded(self):
        """
        Test if the model is loaded.
        """
        response = self.client.get("/is_model_loaded")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"model_loaded": True})
    
    def test_should_get_predict(self):
        """
        Test the predict endpoint with valid input.
        """
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unknown_column_1(self):
        """
        Test the predict endpoint with invalid 'MES' value.
        """
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13  # Invalid month
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)  # Changed from 400 to 422, FastAPI returns 422 when the request is invalid

    def test_should_failed_unknown_column_2(self):
        """
        Test the predict endpoint with invalid 'TIPOVUELO' value.
        """
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O",  # Invalid value
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)  # Changed from 400 to 422, FastAPI returns 422 when the request is invalid
    
    def test_should_failed_unknown_column_3(self):
        """
        Test the predict endpoint with invalid 'OPERA' value.
        """
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas",  # Assuming invalid
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422)  # Changed from 400 to 422, FastAPI returns 422 when the request is invalid