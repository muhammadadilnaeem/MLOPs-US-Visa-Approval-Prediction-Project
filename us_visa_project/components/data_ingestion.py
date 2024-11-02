import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from us_visa_project.entity.config_entity import DataIngestionConfig
from us_visa_project.entity.artifact_entity import DataIngestionArtifact
from us_visa_project.exception import USVISAException
from us_visa_project.logger import logging
from us_visa_project.data_access.usvisa_data import USvisaData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion class with configuration settings.
        
        :param data_ingestion_config: Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            # Raise a custom exception if initialization fails
            raise USVISAException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Exports data from MongoDB into a CSV file as part of the feature store.
        
        Output: Returns the data as a DataFrame.
        On Failure: Logs an error and raises a custom exception.
        """
        try:
            logging.info("Exporting data from MongoDB")
            # Initialize USvisaData to fetch data from MongoDB
            usvisa_data = USvisaData()
            # Export MongoDB collection as a DataFrame
            dataframe = usvisa_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            # Create feature store directory if it doesn't exist and save DataFrame to CSV
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving exported data to feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise USVISAException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the DataFrame into training and testing sets based on the split ratio.
        
        Output: Saves training and testing sets as CSV files.
        On Failure: Logs an error and raises a custom exception.
        """
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            # Perform the train-test split
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train-test split on the DataFrame")

            # Create directory if it doesn't exist and save train/test sets to CSV
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Exporting train and test file paths.")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Exported train and test file paths.")
        except Exception as e:
            raise USVISAException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates data ingestion for the training pipeline, exporting and splitting data.
        
        Output: Returns training and testing data paths as a DataIngestionArtifact.
        On Failure: Logs an error and raises a custom exception.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            # Export data from MongoDB to feature store
            dataframe = self.export_data_into_feature_store()
            logging.info("Retrieved data from MongoDB")

            # Split data into training and testing sets
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train-test split on the dataset")

            # Create an artifact to store paths of training and testing data
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e
