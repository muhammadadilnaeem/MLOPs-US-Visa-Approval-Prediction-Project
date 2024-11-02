
import os
import sys
import pymongo
import certifi
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.constants import DATABASE_NAME, MONGODB_URL_KEY

# Get the CA certificate for secure MongoDB connection
ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   MongoDBClient
    Description :  Provides methods to connect to MongoDB and handle database operations.

    Attributes:
        client : Holds the MongoDB client instance.
        database : Holds the MongoDB database instance.
    
    Output      : Connection to MongoDB database
    On Failure  : Raises an exception if connection fails
    """
    # Static attribute to hold a single instance of the MongoDB client
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        """
        Initializes a MongoDB client and connects to the specified database.

        Parameters:
            database_name (str): Name of the database to connect to.

        Raises:
            USvisaException: Custom exception in case of any connection issues.
        """
        try:
            # Check if the client has already been instantiated
            if MongoDBClient.client is None:
                # Fetch MongoDB URL from environment variables
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    # Raise an error if MongoDB URL is not found in environment variables
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                # Instantiate the MongoDB client with SSL certificate
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            # Assign the client and database instance to the object
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful")
        
        # Handle exceptions by raising a custom exception with additional details
        except Exception as e:
            raise USVISAException(e, sys)
