
# Import necessary modules for system operations, data handling, and machine learning pipelines
import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline

# Import project-specific modules for logging and custom exception handling
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException

# This class maps integer labels to their original string labels
class TargetValueMapping:
    def __init__(self):
        # Initialize mappings for target labels, where 'Certified' is mapped to 0 and 'Denied' to 1
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        # Return the mapping attributes as a dictionary
        return self.__dict__

    def reverse_mapping(self):
        # Generate a reverse mapping dictionary to map integer labels back to their original string labels
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

# This class represents the trained model used to make predictions
class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        Initializes the USvisaModel with preprocessing and model objects.
        :param preprocessing_object: Object used for preprocessing the data (e.g., a scikit-learn Pipeline)
        :param trained_model_object: The trained model used to make predictions
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Transforms the input DataFrame using the preprocessing object and applies the trained model to predict outcomes.
        :param dataframe: Raw input DataFrame
        :return: DataFrame with predictions
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Using the preprocessing object to transform input data")
            # Transform input data to match the training format
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Using the trained model to make predictions")
            # Use the trained model to predict based on transformed features
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            # Raise a custom exception in case of error
            raise USVISAException(e, sys) from e

    def __repr__(self):
        # Represent the model with its class name
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        # String representation of the model object, showing its class name
        return f"{type(self.trained_model_object).__name__}()"
