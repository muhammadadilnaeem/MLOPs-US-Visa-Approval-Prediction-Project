import os
from datetime import date

# Name of the project, can be used for display or logging purposes
PROJECT_NAME = "us_visa_project"

# Database name where the data for the US visa prediction is stored
DATABASE_NAME = "US_VISA_PREDICTION"

# Collection name within the database that holds the US visa data
COLLECTION_NAME = "US_VISA_DATA"

# Environment variable key to access MongoDB URL for database connection
MONGODB_URL_KEY = "MONGODB_URL"

# Name of the pipeline for processing or training steps
PIPELINE_NAME: str = "usvisa"

# Directory where artifacts (models, logs, etc.) will be stored
ARTIFACT_DIR: str = "artifact"

# File name for storing the dataset
FILE_NAME : str = "usvisa.csv"

# File name for storing the training dataset
TRAIN_FILE_NAME : str= "train.csv"

# File name for storing the testing dataset
TEST_FILE_NAME : str = "test.csv"

# File name for storing the trained model
MODEL_FILE_NAME = "model.pkl"


# Name of the target column
TARGET_COLUMN = "case_status"

# Current year for calculating age
CURRENT_YEAR = date.today().year

# Name of the preprocessor file
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# Path to the schema file for validation
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""

# Name of the database collection where the visa data will be stored
DATA_INGESTION_COLLECTION_NAME: str = "US_VISA_DATA"

# Directory name for data ingestion files
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Directory where feature-engineered data will be stored (feature store)
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Directory where ingested (processed) data will be saved
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Ratio for splitting the dataset into training and testing sets
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

# Directory name for data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# File name for drift report
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

# File name that will be used for drift report
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

# Directory name for data transformation
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

# Directory name for transformed data
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

# Directory name for transformed object that will be used for feature engineering
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""

# Directory name for model trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Directory name for trained model
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

# File name for trained model
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

# Expected score for model
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

# File name for model config
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")


# AWS 

# Environment variables
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"

# AWS Secret Access Key
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"

# AWS Region
REGION_NAME = "us-east-1"

"""
MODEL EVALUATION related constant 
"""

# Directory name for model evaluation
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# Bucket name for model
MODEL_BUCKET_NAME = "usvisa-model2024"

# Model registry s3 key
MODEL_PUSHER_S3_KEY = "model-registry"

# Model registry s3 bucket
APP_HOST = "0.0.0.0"

# Model registry s3 bucket
APP_PORT = 8080