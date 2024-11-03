# Import necessary libraries
import os
import sys
import boto3  # AWS SDK for Python
import pickle  # For loading serialized objects

from io import StringIO  # To handle in-memory text streams
from typing import Union, List  # For type hinting
from pandas import DataFrame, read_csv  # For handling dataframes
from us_visa_project.logger import logging  # Custom logging module
from botocore.exceptions import ClientError  # For AWS error handling
from mypy_boto3_s3.service_resource import Bucket  # Type hinting for S3 bucket
from us_visa_project.exception import USVISAException  # Custom exception class
from us_visa_project.configuration.aws_connection import S3Client  # AWS connection configuration


class SimpleStorageService:
    def __init__(self):
        # Initialize the SimpleStorageService with an S3 client and resource instance
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name, s3_key) -> bool:
        # Check if the specified key path exists in the S3 bucket
        try:
            bucket = self.get_bucket(bucket_name)
            # List all objects that match the given s3_key prefix
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0
        except Exception as e:
            raise USVISAException(e, sys)

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """
        Reads an object from S3, optionally decodes it, and returns as a readable object or string.
        """
        logging.info("Entered the read_object method of S3Operations class")

        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode
                else object_name.get()["Body"].read()
            )
            conv_func = lambda: StringIO(func()) if make_readable else func()
            logging.info("Exited the read_object method of S3Operations class")
            return conv_func()
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:
        """
        Retrieves the specified S3 bucket object.
        """
        logging.info("Entered the get_bucket method of S3Operations class")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            logging.info("Exited the get_bucket method of S3Operations class")
            return bucket
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """
        Retrieves the file object from the specified bucket and filename.
        """
        logging.info("Entered the get_file_object method of S3Operations class")

        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
            func = lambda x: x[0] if len(x) == 1 else x
            file_objs = func(file_objects)
            logging.info("Exited the get_file_object method of S3Operations class")
            return file_objs
        except Exception as e:
            raise USVISAException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Loads a serialized model from an S3 bucket.
        """
        logging.info("Entered the load_model method of S3Operations class")

        try:
            model_file = model_name if model_dir is None else f"{model_dir}/{model_name}"
            file_object = self.get_file_object(model_file, bucket_name)
            model_obj = self.read_object(file_object, decode=False)
            model = pickle.loads(model_obj)
            logging.info("Exited the load_model method of S3Operations class")
            return model
        except Exception as e:
            raise USVISAException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """
        Creates a folder in the specified S3 bucket.
        """
        logging.info("Entered the create_folder method of S3Operations class")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                folder_obj = f"{folder_name}/"
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            logging.info("Exited the create_folder method of S3Operations class")

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """
        Uploads a file to the specified S3 bucket and optionally deletes it locally.
        """
        logging.info("Entered the upload_file method of S3Operations class")

        try:
            logging.info(f"Uploading {from_filename} file to {to_filename} in {bucket_name} bucket")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)
            logging.info(f"Uploaded {from_filename} to {to_filename} in {bucket_name} bucket")

            if remove:
                os.remove(from_filename)
                logging.info(f"File {from_filename} deleted after upload.")
            logging.info("Exited the upload_file method of S3Operations class")
        except Exception as e:
            raise USVISAException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        """
        Uploads a DataFrame as a CSV file to an S3 bucket.
        """
        logging.info("Entered the upload_df_as_csv method of S3Operations class")

        try:
            data_frame.to_csv(local_filename, index=None, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
            logging.info("Exited the upload_df_as_csv method of S3Operations class")
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:
        """
        Reads a CSV object from S3 and returns it as a DataFrame.
        """
        logging.info("Entered the get_df_from_object method of S3Operations class")

        try:
            content = self.read_object(object_, make_readable=True)
            df = read_csv(content, na_values="na")
            logging.info("Exited the get_df_from_object method of S3Operations class")
            return df
        except Exception as e:
            raise USVISAException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """
        Reads a CSV file from S3 and returns it as a DataFrame.
        """
        logging.info("Entered the read_csv method of S3Operations class")

        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            df = self.get_df_from_object(csv_obj)
            logging.info("Exited the read_csv method of S3Operations class")
            return df
        except Exception as e:
            raise USVISAException(e, sys) from e
