
# Import necessary modules
import boto3  # AWS SDK for Python
import os  # Module for accessing environment variables
from us_visa_project.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME  # Project-specific constants

class S3Client:
    # Class variables for shared S3 client and resource instances
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """
        Initializes the S3Client class by retrieving AWS credentials from environment variables 
        and establishing a connection to an S3 bucket. Raises an exception if any required 
        environment variables are not set.
        """

        # Check if S3 resource or client has not already been created
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            # Retrieve AWS credentials from environment variables
            __access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            __secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            # Raise an exception if the access key ID environment variable is not set
            if __access_key_id is None:
                raise Exception(f"Environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY} is not set.")

            # Raise an exception if the secret access key environment variable is not set
            if __secret_access_key is None:
                raise Exception(f"Environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set.")

            # Create S3 resource instance for higher-level S3 operations (e.g., object and bucket management)
            S3Client.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name
            )

            # Create S3 client instance for lower-level operations (e.g., bucket and object interaction)
            S3Client.s3_client = boto3.client(
                's3',
                aws_access_key_id=__access_key_id,
                aws_secret_access_key=__secret_access_key,
                region_name=region_name
            )

        # Assign the shared S3 resource and client instances to the instance's attributes
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
