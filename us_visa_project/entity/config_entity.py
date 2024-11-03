
import os
from datetime import datetime
from dataclasses import dataclass
from us_visa_project.constants import *

# Generate a unique timestamp for naming files and directories based on the current date and time
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Configuration for the main training pipeline
@dataclass
class TrainingPipelineConfig:
    # Name of the training pipeline
    pipeline_name: str = PIPELINE_NAME
    # Directory to store artifacts generated during the training process, named with a timestamp
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    # Timestamp string for tracking runs and creating unique directories
    timestamp: str = TIMESTAMP

# Create an instance of the TrainingPipelineConfig to initialize the pipeline configuration
training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

# Configuration for data ingestion settings
@dataclass
class DataIngestionConfig:
    # Directory for data ingestion artifacts, located within the pipeline's artifact directory
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    # File path for storing feature-engineered data (feature store)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    # File path for the training dataset
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    # File path for the testing dataset
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    # Ratio for splitting the data into training and testing sets
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    # Name of the database collection used for storing visa data
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


# Define a class to hold configuration settings for data validation
@dataclass
class DataValidationConfig:
    # Directory path for storing data validation artifacts, set to a subdirectory within the main artifact directory
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    
    # File path for the data drift report, which combines the data validation directory, report directory, and file name
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_DIR,
                                               DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
