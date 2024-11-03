
import sys
from pandas import DataFrame
from us_visa_project.exception import USVISAException
from us_visa_project.entity.target_estimator import USvisaModel
from us_visa_project.cloud_storage.aws_storage import SimpleStorageService

class USvisaEstimator:
    """
    This class is used to save and retrieve US visa models in an S3 bucket and to perform predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Initializes the USvisaEstimator instance.

        :param bucket_name: Name of the S3 bucket where the model is stored.
        :param model_path: Path to the model file within the S3 bucket.
        """
        self.bucket_name = bucket_name  # Store the name of the S3 bucket
        self.s3 = SimpleStorageService()  # Create an instance of SimpleStorageService to interact with S3
        self.model_path = model_path  # Store the model path within the S3 bucket
        self.loaded_model: USvisaModel = None  # Placeholder for the loaded model instance

    def is_model_present(self, model_path: str) -> bool:
        """
        Check if the model exists in the specified S3 bucket.

        :param model_path: Path to the model in the S3 bucket.
        :return: True if the model is present, otherwise False.
        """
        try:
            # Check if the S3 key path for the model is available
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except USVISAException as e:
            print(e)  # Print the exception message if an error occurs
            return False  # Return False if an exception is caught

    def load_model(self) -> USvisaModel:
        """
        Load the model from the specified S3 path.

        :return: The loaded USvisaModel instance.
        """
        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)  # Load the model using the S3 service

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save the model to the specified S3 path.

        :param from_file: Local path of the model to be saved.
        :param remove: If True, the local model file will be deleted after upload; default is False.
        :return: None
        """
        try:
            # Upload the model file to S3 and optionally remove the local file
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            raise USVISAException(e, sys)  # Raise a custom exception if an error occurs

    def predict(self, dataframe: DataFrame) -> any:
        """
        Perform predictions using the loaded model.

        :param dataframe: Input DataFrame for prediction.
        :return: Predictions made by the loaded model.
        """
        try:
            # Load the model if it has not been loaded yet
            if self.loaded_model is None:
                self.loaded_model = self.load_model()  # Load the model from S3

            return self.loaded_model.predict(dataframe=dataframe)  # Use the model to make predictions
        except Exception as e:
            raise USVISAException(e, sys)  # Raise a custom exception if an error occurs
