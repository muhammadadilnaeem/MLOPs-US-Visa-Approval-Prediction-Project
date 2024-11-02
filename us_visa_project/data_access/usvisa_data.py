
import sys
import numpy as np
import pandas as pd
from typing import Optional

from us_visa_project.constants import DATABASE_NAME
from us_visa_project.exception import USVISAException
from us_visa_project.configuration.mongo_db_connection import MongoDBClient


class USvisaData:
    """
    A class to export records from MongoDB collections as a pandas DataFrame.
    """

    def __init__(self):
        """
        Initializes the MongoDB client for accessing the database.
        """
        try:
            # Initialize MongoDB client with the specified database name
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            # Raise a custom exception if the MongoDB client initialization fails
            raise USVISAException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
            collection_name (str): Name of the MongoDB collection to export.
            database_name (Optional[str]): Name of the database (optional, defaults to DATABASE_NAME).

        Returns:
            pd.DataFrame: DataFrame containing all records from the specified collection.
        
        Raises:
            USvisaException: Custom exception for handling any issues during export.
        """
        try:
            # Access the specified collection within the default or given database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Convert the collection records to a DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Drop the "_id" column if it exists in the DataFrame
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            # Replace any "na" values in the DataFrame with NaN for consistency
            df.replace({"na": np.nan}, inplace=True)
            
            return df

        except Exception as e:
            # Raise a custom exception if there's an error during the export process
            raise USVISAException(e, sys)
