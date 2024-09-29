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
    
    def test_should_get_predict_0(self):
        """
        Test the predict endpoint with valid input that should return a prediction of 0.
        """
        data = {
            "flights": [
                {
                    "OPERA": "Copa Air",  # Valid value from top features
                    "TIPOVUELO": "I",  # Valid value from top features
                    "MES": 4  # Valid value from top features
                }
            ]
        }
        # Uncomment and update the model if you want to mock the predict method
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # Mocking prediction to return 0
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    
    def test_should_get_predict_1(self):
        """
        Test the predict endpoint with valid input that should return a prediction of 1.
        """
        data = {
            "flights": [
                {
                    "OPERA": "Latin American Wings",  # Valid value from top features
                    "TIPOVUELO": "I",  # Valid value from top features
                    "MES": 7  # Valid value from top features
                }
            ]
        }
        # Uncomment and update the model if you want to mock the predict method
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([1])) # Mocking prediction to return 1
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [1]})

    def test_should_failed_unknown_column_1(self):
        """
        Test the predict endpoint with invalid 'OPERA' value.
        """
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", # Invalid
                    "TIPOVUELO": "I",  # Valid value from top features
                    "MES": 7  # Valid value from top features
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unknown_column_2(self):
        """
        Test the predict endpoint with invalid 'TIPOVUELO' value.
        """
        data = {        
            "flights": [
                {
                    "OPERA": "Latin American Wings",  # Valid value from top features
                    "TIPOVUELO": "O",  # Invalid
                    "MES": 7  # Valid value from top features
                }
            ]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
        
    def test_should_failed_unknown_column_3(self):
        """
        Test the predict endpoint with invalid 'MES' value.
        """
        data = {        
            "flights": [
                {
                    "OPERA": "Latin American Wings",  # Valid value from top features
                    "TIPOVUELO": "I",  # Valid value from top features
                    "MES": 13  # Invalid
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 422) # FastAPI returns 422 for invalid data