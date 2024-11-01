# Import necessary modules
import os
import sys
import dill  # For serializing and deserializing Python objects
import yaml  # For reading and writing YAML files
from pandas import DataFrame  # For handling data in DataFrame format
import numpy as np  # For handling numerical operations with arrays

from us_visa_project.exception import USvisaException  # Custom exception for error handling
from us_visa_project.logger import logging  # Custom logging for tracking execution

# Function to read and parse a YAML file
def read_yaml_file(file_path: str) -> dict:
    try:
        # Open and safely load the YAML file into a dictionary
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    # Catch any exceptions, raise a custom USvisaException with system details
    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to write data to a YAML file
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        # Replace the file if it already exists and replace is set to True
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        # Create the directory if it doesn’t exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write content to the YAML file
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to load a serialized Python object from a file
def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:
        # Open and load the serialized object from the file
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to save a numpy array to a file
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to a specified file path
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        # Ensure the directory exists, then save the numpy array
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to load a numpy array from a file
def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from a specified file path
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        # Open and load the numpy array from the file
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to save a Python object using dill serialization
def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        # Ensure the directory exists, then save the object using dill
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e


# Function to drop specified columns from a pandas DataFrame
def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop specified columns from a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns method of utils")

    try:
        # Drop the specified columns from the DataFrame
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")

        return df
    except Exception as e:
        raise USvisaException(e, sys) from e
