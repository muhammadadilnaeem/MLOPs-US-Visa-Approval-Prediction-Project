# Import necessary libraries for JSON handling, system functions, and data manipulation
import json
import sys
import pandas as pd
from pandas import DataFrame

# Import Evidently library components for data drift analysis
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

# Import project-specific modules for logging, exceptions, utility functions, and entity configurations
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.constants import SCHEMA_FILE_PATH
from us_visa_project.entity.config_entity import DataValidationConfig
from us_visa_project.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa_project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact




class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Constructor initializes data validation with artifacts and configuration.
        :param data_ingestion_artifact: Reference to data ingestion output artifact
        :param data_validation_config: Configuration settings for data validation
        """
        try:
            # Store references to the data ingestion artifact and validation configuration
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            # Load schema configuration file containing column specifications
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            # Raise a custom exception in case of error
            raise USVISAException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates that the DataFrame has the expected number of columns.
        :param dataframe: DataFrame to validate
        :return: True if column count matches expected; False otherwise
        """
        try:
            # Check if the number of columns matches schema configuration
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USVISAException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validates the existence of both numerical and categorical columns as defined in the schema.
        :param df: DataFrame to validate
        :return: True if all required columns exist; False otherwise
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []

            # Check for missing numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if missing_numerical_columns:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            # Check for missing categorical columns
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            if missing_categorical_columns:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            # Return False if any columns are missing
            return not (missing_categorical_columns or missing_numerical_columns)
        except Exception as e:
            raise USVISAException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Reads data from a CSV file into a DataFrame.
        :param file_path: Path to the CSV file
        :return: DataFrame with loaded data
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USVISAException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects data drift between two DataFrames (reference and current).
        :param reference_df: Reference DataFrame (e.g., training data)
        :param current_df: Current DataFrame (e.g., new data)
        :return: True if data drift is detected; False otherwise
        """
        try:
            # Initialize data drift profile from Evidently
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])
            data_drift_profile.calculate(reference_df, current_df)

            # Parse and save the drift report as JSON
            report = data_drift_profile.json()
            json_report = json.loads(report)
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            # Extract drift metrics
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise USVISAException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process, checks column requirements, and detects data drift.
        :return: DataValidationArtifact with the results of validation
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            # Load training and testing data
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            # Validate column count for training data
            status = self.validate_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe."
            
            # Validate column count for testing data
            status = self.validate_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in test dataframe."

            # Validate column existence for training data
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe."
            
            # Validate column existence for testing data
            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe."

            # Determine overall validation status based on error message presence
            validation_status = len(validation_error_msg) == 0

            # If validation passes, check for data drift
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                validation_error_msg = "Drift detected" if drift_status else "Drift not detected"
            else:
                logging.info(f"Validation error: {validation_error_msg}")
                
            # Create artifact for validation results
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e
