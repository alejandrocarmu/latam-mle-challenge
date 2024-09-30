# Challenge Documentation

This document details the steps I took to complete the provided challenge, broken down into four parts. Each part outlines the work done, the choices made, and the results obtained.

---

## Part I: Transcribing and Operationalizing the Model

In this section, I transcribed the provided `.ipynb` file into the `model.py` file and implemented several improvements to ensure the code follows best practices and adheres to the project requirements.

### Plotting Fixes

I fixed issues with `sns.barplot` calls by explicitly adding the `x` and `y` keyword arguments. For example, I changed:

    sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)
    

to:

    sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)

Additionally, I updated the style settings for the plots by using:

    sns.set_theme(style="darkgrid")

This change replaced the outdated `sns.set()` call to improve the plot's visual consistency.

### Model Feature Selection

The following top 10 features were selected for the model based on feature importance:

    top_10_features = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

**IMPORTANT NOTE**:
In order to ensure that everything works properly with the top features provided, I had to adjust the API and the test scripts (`test_model.py` and `test_api.py`) to comply with the model and its expectations. This included adjusting the input handling and ensuring the tests align with the model's feature set.

### **Model Analysis and Conclusion**

#### Summary of Results

- **Class balancing significantly improved recall for the delay class (1)**, particularly in XGBoost and Logistic Regression models.
- **Reducing the feature set to the top 10** did not negatively impact model performance, as observed in Models 3 to 6.
- **XGBoost (Model 5)** showed a slight advantage in terms of recall for the delay class over Logistic Regression.

#### Best Model

The best-performing model is **Model 5: XGBoost with Top 10 Features and Class Balancing**, which:

- Achieved a **recall of 0.69** for the delay class (1), critical for identifying flight delays.
- Showed satisfactory precision for class 0 (No Delay).

#### Future Improvements

- **Low precision for the delay class** can be improved through techniques like **SMOTE** or by further tuning the **scale_pos_weight** in XGBoost.
- Additional advanced feature engineering and hyperparameter tuning should be explored in future iterations.

### Code Implementation

The model's implementation in `model.py` includes several important methods, each with a specific role in ensuring the functionality of the model:

- **`preprocess(data: pd.DataFrame)`**: This method processes raw data to prepare it for training or prediction. It handles datetime conversion, feature engineering (e.g., calculating the period of the day, determining high season, and computing time differences), and one-hot encoding for categorical features like `OPERA`, `TIPOVUELO`, and `MES`. The method ensures the consistency of features during prediction by storing and reusing category mappings from training time.

- **`fit(features: pd.DataFrame, target: pd.Series)`**: This method trains the XGBoost model using class balancing. It shuffles the data, splits it into training and testing sets, and calculates the `scale_pos_weight` to address class imbalance. The trained model is saved to disk for later use.

- **`predict(features: pd.DataFrame)`**: This method loads the pre-trained model (if not already loaded) and predicts flight delays based on the input features. It ensures that the features match the expected structure before making predictions.

- **`load_model()`**: This method loads the pre-trained model and its associated feature names from the disk.

- **`api_preprocess(data: pd.DataFrame)`**: A simplified version of the preprocessing method, designed for API use. It focuses only on `OPERA`, `TIPOVUELO`, and `MES` features and ensures consistency with the top features selected for the model.

The model includes preprocessing functions for feature engineering, a method for model training with class balancing, and a prediction function using the XGBoost model.

### Tests

I adjusted the provided test scripts in `test_model.py` to ensure that the model processes data correctly, fits the model as expected, and produces predictions that comply with the performance metrics. An important note is that the tests were designed to meet the requirements of precision and recall for both classes.

The following assertions were used in the model tests to ensure the expected behavior of the model:

    assert report["0"]["recall"] < 0.60
    assert report["0"]["f1-score"] < 0.70
    assert report["1"]["recall"] > 0.60
    assert report["1"]["f1-score"] > 0.30

This ensures that the model meets the necessary benchmarks for flight delay prediction.

### Important Notes

- The model and API tests were tailored to ensure compliance with the provided top features and the performance metrics defined above.
- I also updated the libraries used in this implementation to prevent deprecation warnings and improve stability.

Once the model was fully transcribed and tested, it successfully passed the tests by running:

    make model-test

This ensures the model is ready for integration into the next steps of the challenge.

---

## Part II: API Development with FastAPI

In this part, I deployed the model via an API using FastAPI. The `api.py` file handles model inference by accepting flight information in a structured format, pre-processing the input, and predicting potential flight delays based on the trained model.

### **API Implementation**

The FastAPI implementation includes the following key components:

1. **Lifespan Management**

   The `lifespan` context manager is used to handle startup and shutdown logic for the API. This ensures that the model is loaded when the API starts and allows for a graceful shutdown when the API is stopped. During startup, the `DelayModel` class is instantiated, and the model is loaded from the saved files (`best_model.pkl` and `feature_names.pkl`).

2. **Prediction Endpoint**
   The `/predict` endpoint accepts flight information in the form of `OPERA`, `TIPOVUELO`, and `MES` using the `Flight` Pydantic model. The input data is pre-processed, and predictions are made using the pre-trained XGBoost model. The response is a list of predicted classes (0 or 1) indicating whether a flight is expected to be delayed.

3. **Health and Root Endpoints**
   The root endpoint (`/`) returns a simple welcome message, while the `/health` endpoint serves as a health check to verify the API’s status.

4. **Input Validation**
   The Pydantic models validate the input data structure, including ensuring that the `MES` field is between 1 and 12. This validation happens automatically before data reaches the prediction logic, ensuring that incorrect inputs return a `422 Unprocessable Entity` error, handled by FastAPI.

### **Tests**

The `test_api.py` script tests various aspects of the API:

1. **Root and Health Endpoints**:
   - The root endpoint (`/`) and the health endpoint (`/health`) are tested to ensure the API is running and can return expected responses.

2. **Model Prediction**:
   - Two tests are included for the `/predict` endpoint, each testing valid input data:
     - **Test 1**: Predicting with `OPERA` as `Copa Air` and `MES` as `4`, which should return a prediction of `0`.
     - **Test 2**: Predicting with `OPERA` as `Latin American Wings` and `MES` as `7`, which should return a prediction of `1`.

3. **Invalid Input Handling**:
   - Several tests validate the behavior of the API when incorrect or unknown inputs are provided:
     - **Invalid `OPERA`**: When an unrecognized airline (`Aerolineas Argentinas`) is used, the test expects a `400 Bad Request` response.
     - **Invalid `TIPOVUELO`**: When an invalid `TIPOVUELO` is provided, the API returns a `400 Bad Request`.
     - **Invalid `MES`**: Since the `MES` column has been validated using Pydantic with values between 1 and 12, any value outside of this range triggers a `422 Unprocessable Entity` error, not a `400`.

### **Important Note**

In this part, I adjusted the test data to work properly with the model's top 10 features as defined by the Data Science team. These top features include airlines such as `Copa Air` and `Latin American Wings`, as well as valid values for `TIPOVUELO` and `MES`.

I also ensured that the Pydantic model enforces input validation for the `MES` column, restricting it to values between 1 and 12. This validation ensures that any request with an invalid month will return a `422` error (handled by FastAPI), which is more appropriate than the originally suggested `400` error.

Once the API was fully implemented, I ran the tests using:

    make api-test

This confirmed that the API correctly processes input, makes predictions, and handles errors as expected, all while passing the tests.

---

## Part III: Deploying the API to Google Cloud

For deployment, I chose **Google Cloud Run**. Here's how I went about deploying the API:

1. **Containerization**: I built the Docker image for the FastAPI service and pushed it to Google Container Registry (GCR). The Dockerfile was designed to optimize for image size and speed by using the Python slim image and pip caching.

2. **Deployment Process**
   - I used `gcloud` commands to deploy the API to Cloud Run.
   - I chose `southamerica-east1` as the region, given that it’s the nearest to my location in Colombia, and also covers the company's location in Chile.

3. **Stress Testing**: After deployment, I verified the API by running stress tests using `locust`. The stress test simulated concurrent users making requests to the `/predict` endpoint to ensure the API could handle the expected load. The API successfully passed the tests, showing it could handle up to 100 users.

    make stress-test

---

## Part IV: CI/CD Implementation

For the CI/CD pipeline, I used **GitHub Actions** to automate the testing and deployment of the API.

1. **Continuous Integration (CI)**: The `ci.yml` file was configured to:
   - Run tests whenever there is a push or pull request to the `develop` branch.
   - Install dependencies, run the model and API tests, and upload the results.

2. **Continuous Delivery (CD)**: The `cd.yml` file was configured to:
   - Deploy the API to Google Cloud Run when changes are merged into the `main` branch.
   - The `cd.yml` file uses the service account's credentials to authenticate and deploy the container.

The CI/CD pipeline ensures that all code is tested and deployed automatically, minimizing the chance of errors during deployment.

To ensure the secrets were used properly, I configured the following GitHub secret:

- `GCP_CREDENTIALS`: The service account key in JSON format for deploying to Google Cloud Run.

---
