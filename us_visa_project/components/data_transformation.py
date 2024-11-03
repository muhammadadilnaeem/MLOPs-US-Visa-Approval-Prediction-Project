
# Import necessary modules
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer

# Import project-specific modules for logging, custom exceptions, and utilities
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.entity.target_estimator import TargetValueMapping
from us_visa_project.entity.config_entity import DataTransformationConfig
from us_visa_project.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from us_visa_project.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from us_visa_project.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        Initialize the DataTransformation class with data ingestion artifacts, configuration, and validation status.
        :param data_ingestion_artifact: Reference to the artifact from the data ingestion stage
        :param data_transformation_config: Configuration settings for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            # Load schema configuration for data transformation
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USVISAException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads data from a CSV file and returns it as a DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USVISAException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates a preprocessing pipeline for data transformation.
        :return: A configured ColumnTransformer object for data processing
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Initialize transformers for numerical, one-hot, and ordinal columns
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            logging.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            # Retrieve column names for each transformation from schema configuration
            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            # Power transformation for specified columns
            transform_pipe = Pipeline(steps=[('transformer', PowerTransformer(method='yeo-johnson'))])

            # Define ColumnTransformer to apply different transformations on different column groups
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return preprocessor

        except Exception as e:
            raise USVISAException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates data transformation, applying preprocessing and handling imbalances in the data.
        :return: DataTransformationArtifact containing paths to the transformed data and preprocessing object
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                # Get the data transformer/preprocessor
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                # Load training and testing data
                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Separate input and target features for the training data
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Separated train features and target column for training dataset")

                # Add calculated 'company_age' column
                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                logging.info("Added company_age column to the training dataset")

                # Drop specified columns according to schema configuration
                drop_cols = self._schema_config['drop_columns']
                input_feature_train_df = drop_columns(df=input_feature_train_df, cols=drop_cols)

                # Map target values to integers as per TargetValueMapping
                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())

                # Separate input and target features for the testing data
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                # Add calculated 'company_age' column to the test data
                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']
                logging.info("Added company_age column to the test dataset")

                # Drop specified columns in test data
                input_feature_test_df = drop_columns(df=input_feature_test_df, cols=drop_cols)
                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

                logging.info("Separated test features and target column for testing dataset")

                # Apply transformations to training and testing input features
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # Use SMOTEENN to handle class imbalance in training and testing data
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)

                # Combine transformed features and target for training and testing data
                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                # Save the preprocessor object and transformed data arrays
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object and transformed data arrays")

                # Create and return DataTransformationArtifact with paths to transformed data
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact

            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USVISAException(e, sys) from e
