import os  # Importing the os module for operating system related functionality
from pathlib import Path  # Importing Path class from pathlib module for handling filesystem paths

# Define the name of the project
project_name = "us_visa_project"

# List of files and directories to be created for the project
list_of_files = [
    f"{project_name}/__init__.py",  # Package initialization file for the project
    f"{project_name}/components/__init__.py",  # Package initialization for components
    f"{project_name}/components/data_ingestion.py",  # File for data ingestion logic
    f"{project_name}/components/data_validation.py",  # File for data validation logic
    f"{project_name}/components/data_transformation.py",  # File for data transformation logic
    f"{project_name}/components/model_trainer.py",  # File for model training logic
    f"{project_name}/components/model_evaluation.py",  # File for model evaluation logic
    f"{project_name}/components/model_pusher.py",  # File for pushing models to production
    f"{project_name}/configuration/__init__.py",  # Package initialization for configuration
    f"{project_name}/constants/__init__.py",  # Package initialization for constants
    f"{project_name}/entity/__init__.py",  # Package initialization for entities
    f"{project_name}/entity/config_entity.py",  # File for configuration entity definitions
    f"{project_name}/entity/artifact_entity.py",  # File for artifact entity definitions
    f"{project_name}/exception/__init__.py",  # Package initialization for exceptions
    f"{project_name}/logger/__init__.py",  # Package initialization for logging functionality
    f"{project_name}/pipline/__init__.py",  # Package initialization for pipeline
    f"{project_name}/pipline/training_pipeline.py",  # File for training pipeline logic
    f"{project_name}/pipline/prediction_pipeline.py",  # File for prediction pipeline logic
    f"{project_name}/utils/__init__.py",  # Package initialization for utility functions
    f"{project_name}/utils/main_utils.py",  # File for main utility functions
    "app.py",  # Main application file
    "requirements.txt",  # File listing project dependencies
    "Dockerfile",  # Docker configuration file
    ".dockerignore",  # Specifies files to ignore during Docker build
    "demo.py",  # Demo script for showcasing functionality
    "setup.py",  # Setup script for packaging the project
    "config/model.yaml",  # Configuration file for model parameters
    "config/schema.yaml",  # Schema definition file for data
]

# Loop through the list of files
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert string filepath to Path object
    filedir, filename = os.path.split(filepath)  # Split the filepath into directory and filename
    if filedir != "":  # Check if the directory part is not empty
        os.makedirs(filedir, exist_ok=True)  # Create the directory if it does not exist
    # Check if the file does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:  # Create the file (or open if it exists)
            pass  # Do nothing after creating the file
    else:
        print(f"file is already present at: {filepath}")  # Inform the user that the file already exists