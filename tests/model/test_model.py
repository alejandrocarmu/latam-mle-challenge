import unittest
import pandas as pd
import pathlib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
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

    TARGET_COL = [
        "delay"
    ]


    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        # Fix for relative path
        # Use pathlib to resolve the path to data.csv
        data_path = pathlib.Path(__file__).parent.parent.parent / "data" / "data.csv"
        
        # Load the data
        self.data = pd.read_csv(filepath_or_buffer=data_path)
        

    def test_model_preprocess_for_training(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)


    def test_model_preprocess_for_serving(
        self
    ):
        features = self.model.preprocess(
            data=self.data
        )

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)


    def test_model_fit(
        self
    ):
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        _, features_validation, _, target_validation = train_test_split(features, target, test_size = 0.33, random_state = 42)

        self.model.fit(
            features=features,
            target=target
        )

        predicted_target = self.model._model.predict(
            features_validation
        )

        report = classification_report(target_validation, predicted_target, output_dict=True)
        
        assert report["0"]["recall"] < 0.60
        assert report["0"]["f1-score"] < 0.70
        assert report["1"]["recall"] > 0.60
        assert report["1"]["f1-score"] > 0.30


    def test_model_predict(
        self
    ):
        # Preprocess the data
        features, target = self.model.preprocess(
            data=self.data,
            target_column="delay"
        )

        # Split the data into training and testing sets
        x_train, x_test, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)

        # Train the model before prediction
        self.model.fit(
            features=x_train,
            target=y_train
        )

        # Now predict the targets using the preprocessed test set
        predicted_targets = self.model.predict(
            features=x_test  # These features are already preprocessed, no need to preprocess again
        )

        # Assertions to validate the predictions
        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == x_test.shape[0]
        assert all(isinstance(predicted_target, int) for predicted_target in predicted_targets)

