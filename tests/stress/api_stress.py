from locust import HttpUser, task

class StressUser(HttpUser):
    @task
    def predict_latam(self):
        """
        Test using a valid 'OPERA' value 'Grupo LATAM' and a valid 'TIPOVUELO' value 'I'.
        """
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM",  # Valid value from top features
                        "TIPOVUELO": "I",        # Valid value (International)
                        "MES": 10                # Valid value for 'MES' from top features
                    }
                ]
            }
        )

    @task
    def predict_copa(self):
        """
        Test using another valid 'OPERA' value 'Copa Air' and a valid 'TIPOVUELO' value 'I'.
        """
        self.client.post(
            "/predict", 
            json={
                "flights": [
                    {
                        "OPERA": "Copa Air",     # Valid value from top features
                        "TIPOVUELO": "I",        # Valid value (International)
                        "MES": 12                # Valid 'MES' value from top features
                    }
                ]
            }
        )
