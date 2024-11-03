
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


# Define a configuration class for data transformation settings
@dataclass
class DataTransformationConfig:
    # Directory for storing data transformation artifacts, set to a subdirectory within the main artifact directory
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    
    # File path for the transformed training data, saved as a .npy (NumPy) file for efficiency
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    
    # File path for the transformed testing data, also saved as a .npy file
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    
    # File path for the preprocessing object (e.g., scaler or encoder), used to apply consistent transformations
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)


# Define a class to hold configuration settings for model training
@dataclass
class ModelTrainerConfig:
    # Directory where the model training artifacts will be stored
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    
    # File path where the trained model will be saved
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    
    # The minimum expected accuracy score for the model to be considered satisfactory
    expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    
    # File path for the model configuration file, which contains model-specific parameters for training
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

# Define a class to hold configuration settings for model evaluation
@dataclass
class ModelEvaluationConfig:
    """
    Configuration for evaluating a model, including thresholds and S3 bucket details.
    """
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE  # Threshold score for detecting significant model change
    bucket_name: str = MODEL_BUCKET_NAME  # Name of the S3 bucket where the model is stored
    s3_model_key_path: str = MODEL_FILE_NAME  # Path within the S3 bucket to the model file

@dataclass
class ModelPusherConfig:
    """
    Configuration for pushing a model to S3 storage.
    """
    bucket_name: str = MODEL_BUCKET_NAME  # Name of the S3 bucket where the model will be pushed
    s3_model_key_path: str = MODEL_FILE_NAME  # Path within the S3 bucket for saving the model file

@dataclass
class USvisaPredictorConfig:
    """
    Configuration for the US visa prediction model, including file paths and S3 bucket details.
    """
    model_file_path: str = MODEL_FILE_NAME  # Local file path of the model used for prediction
    model_bucket_name: str = MODEL_BUCKET_NAME  # Name of the S3 bucket containing the model for predictions
