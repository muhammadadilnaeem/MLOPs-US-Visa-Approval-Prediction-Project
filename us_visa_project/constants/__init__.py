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


# Constants related to data ingestion configuration

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

