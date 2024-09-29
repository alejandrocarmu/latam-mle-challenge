import os
import pandas as pd
import numpy as np
import joblib
from typing import Tuple, Union, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import logging
from rich.logging import RichHandler

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

logger = logging.getLogger("DelayModelLogger")

class DelayModel:
    def __init__(self):
        """
        Initialize the DelayModel with necessary attributes.
        """
        self._model = None  # Model should be saved in this attribute.
        self.top_features = [
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

        # Store one-hot encoding categories to ensure consistency during prediction
        self.opera_categories = None
        self.tipovuelo_categories = None
        self.mes_categories = None

        # Define directory for model and feature files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, "models")
        self.model_path = os.path.join(self.model_dir, "best_model.pkl")
        self.features_path = os.path.join(self.model_dir, "feature_names.pkl")

        # Ensure the models/ directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.
        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If set, the target is returned.
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target.
            or
            pd.DataFrame: Features.
        """
        logger.info("Starting preprocessing of data.")

        # Convert 'Fecha-I' and 'Fecha-O' to datetime
        for col in ['Fecha-I', 'Fecha-O']:
            data[col] = pd.to_datetime(data[col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            logger.info(f"Converted column '{col}' to datetime.")

        # Drop rows with missing critical dates
        initial_shape = data.shape
        data = data.dropna(subset=['Fecha-I', 'Fecha-O']).reset_index(drop=True)
        logger.info(f"Dropped rows with missing dates: {initial_shape} -> {data.shape}")

        # Feature Engineering
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
        logger.info("Generated 'period_day' feature.")

        data['high_season'] = self.is_high_season_vectorized(data['Fecha-I'])
        logger.info("Generated 'high_season' feature.")

        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
        data.loc[data['min_diff'] < 0, 'min_diff'] = np.nan  # Handle negative differences
        logger.info("Calculated 'min_diff' feature and handled negative differences.")

        if target_column:
            data['delay'] = (data['min_diff'] > 15).astype(int)
            logger.info("Generated 'delay' target variable.")

        # One-Hot Encoding for 'OPERA', 'TIPOVUELO', 'MES'
        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
        logger.info("Performed one-hot encoding for 'OPERA', 'TIPOVUELO', and 'MES'.")

        # Store categories during training for consistent encoding during prediction
        if target_column and self.opera_categories is None:
            self.opera_categories = opera_dummies.columns
            self.tipovuelo_categories = tipovuelo_dummies.columns
            self.mes_categories = mes_dummies.columns
            logger.info("Stored one-hot encoding categories for future predictions.")
        elif not target_column:
            # During prediction, ensure the same columns are present
            opera_dummies = opera_dummies.reindex(columns=self.opera_categories, fill_value=0)
            tipovuelo_dummies = tipovuelo_dummies.reindex(columns=self.tipovuelo_categories, fill_value=0)
            mes_dummies = mes_dummies.reindex(columns=self.mes_categories, fill_value=0)
            logger.info("Reindexed one-hot encoded features to match training categories.")

        # Concatenate all dummy variables
        encoded_features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)
        logger.info("Concatenated one-hot encoded features.")

        # Select top 10 features and ensure all expected columns are present
        try:
            features = encoded_features.reindex(columns=self.top_features, fill_value=0)
            logger.info("Selected top 10 features, filling missing columns with zeros.")
        except KeyError as e:
            missing_cols = set(self.top_features) - set(encoded_features.columns)
            logger.error(f"Missing columns in the data: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        if target_column:
            target = data[[target_column]]  # Use double brackets to ensure it's a DataFrame
            logger.info("Preprocessing completed. Returning features and target.")
            return features, target
        else:
            logger.info("Preprocessing completed. Returning features.")
            return features

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Fit the XGBoost model with preprocessed data.
        Args:
            features (pd.DataFrame): Preprocessed data.
            target (pd.Series): Target variable.
        """
        logger.info("Starting model training.")

        # Shuffle the data
        features, target = shuffle(features, target, random_state=111)
        logger.info("Shuffled the data.")

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        logger.info(f"Split data into training and testing sets: {x_train.shape} | {x_test.shape}")

        # Calculate scale_pos_weight for XGBoost to handle class imbalance
        n_y0 = y_train.value_counts().get(0, 0)
        n_y1 = y_train.value_counts().get(1, 0)
        scale_pos_weight = n_y0 / n_y1 if n_y1 != 0 else 1
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight}")

        # Initialize and train XGBoost model
        self._model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )
        self._model.fit(x_train, y_train)
        logger.info("Model training completed.")

        # Evaluate the model on test data
        y_preds = self._model.predict(x_test)
        cm = confusion_matrix(y_test, y_preds)
        logger.info("Confusion Matrix:")
        logger.info(cm)

        report = classification_report(y_test, y_preds)
        logger.info("Classification Report:")
        logger.info(report)

        # Save the trained model and feature names
        joblib.dump(self._model, self.model_path)
        joblib.dump(self.top_features, self.features_path)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Feature names saved to {self.features_path}")

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.
        
        Args:
            features (pd.DataFrame): Preprocessed data.

        Returns:
            List[int]: Predicted targets.
        """
        if self._model is None:
            logger.info("Loading model and feature names.")
            # Load the model
            self._model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")

            # Load the feature names
            self.top_features = joblib.load(self.features_path)
            logger.info(f"Feature names loaded from {self.features_path}")

        # Ensure the features are reindexed properly to include all expected columns
        try:
            processed_features = features.reindex(columns=self.top_features, fill_value=0)
        except KeyError as e:
            logger.error(f"Missing columns in input features: {e}")
            raise KeyError(f"Missing columns: {e}")

        # Prediction logic
        logger.info("Starting model prediction.")
        predictions = self._model.predict(processed_features)
        
        # Convert predictions to list of integers
        predictions = predictions.tolist()
        predictions = [int(pred) for pred in predictions]
        
        logger.info("Prediction completed.")
        return predictions

    def load_model(self):
        """
        Load the trained model and feature names from the models directory.
        """
        logger.info(f"Loading model from: {self.model_path}")
        logger.info(f"Loading feature names from: {self.features_path}")
        if not os.path.exists(self.model_path):
            logger.error(f"Model file {self.model_path} not found.")
            raise FileNotFoundError("Model file not found.")
        if not os.path.exists(self.features_path):
            logger.error(f"Feature names file {self.features_path} not found.")
            raise FileNotFoundError("Feature names file not found.")

        self._model = joblib.load(self.model_path)
        self.top_features = joblib.load(self.features_path)
        logger.info(f"Model loaded from {self.model_path}")
        logger.info(f"Feature names loaded from {self.features_path}")

    @staticmethod
    def get_period_day(date_time: datetime) -> str:
        """
        Categorize the time of day into periods.

        Args:
            date_time (datetime): Datetime object.

        Returns:
            str: 'mañana', 'tarde', or 'noche'.
        """
        time = date_time.time()
        if datetime.strptime("05:00", '%H:%M').time() <= time <= datetime.strptime("11:59", '%H:%M').time():
            return 'mañana'
        elif datetime.strptime("12:00", '%H:%M').time() <= time <= datetime.strptime("18:59", '%H:%M').time():
            return 'tarde'
        else:
            return 'noche'

    def is_high_season_vectorized(self, fecha_series: pd.Series) -> pd.Series:
        """
        Determine if a flight is in high season based on 'Fecha-I'.

        Args:
            fecha_series (pd.Series): Series containing datetime objects.

        Returns:
            pd.Series: Series containing 1 for high season and 0 otherwise.
        """
        month = fecha_series.dt.month
        day = fecha_series.dt.day

        condition1 = ((month == 12) & (day >= 15)) | \
                     ((month == 1) & (day <= 31)) | \
                     ((month == 2) & (day <= 29)) | \
                     ((month == 3) & (day <= 3))
        
        condition2 = ((month == 7) & (day >= 15)) | \
                     ((month == 8) & (day <= 31)) | \
                     ((month == 9) & (day <= 11))
        
        condition3 = ((month == 9) & (day >= 11)) | \
                     ((month == 10) & (day <= 30))
        
        high_season = condition1 | condition2 | condition3
        logger.info("Determined high season flags.")
        return high_season.astype(int)

    def api_preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified preprocessing for the API that only deals with 'OPERA', 'TIPOVUELO', and 'MES'.
        
        Args:
            data (pd.DataFrame): Raw data.

        Returns:
            pd.DataFrame: Features.
        """
        logger.info("Starting simplified preprocessing for API.")

        # One-Hot Encoding for 'OPERA', 'TIPOVUELO', 'MES'
        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
        logger.info("Performed one-hot encoding for 'OPERA', 'TIPOVUELO', and 'MES'.")

        # Reindex to match top_features, fill missing columns with 0
        features = pd.concat([opera_dummies, tipovuelo_dummies, mes_dummies], axis=1)
        features = features.reindex(columns=self.top_features, fill_value=0)
        logger.info("Reindexed features to match top_features.")

        # Convert boolean columns to integers to ensure consistency
        features = features.astype(int)
        logger.info("Converted features to integer type.")

        logger.info("Simplified preprocessing completed. Returning features.")
        return features
