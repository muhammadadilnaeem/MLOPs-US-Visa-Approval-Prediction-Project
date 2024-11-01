# Import necessary modules
import os
import sys  # For handling system-specific parameters and functions

# Function to extract and format error details
def message_details(error, error_detail: sys):

    # Extract traceback information for the exception
    _, _, exc_tb = error_detail.exc_info()

    # Retrieve the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message with the file name, line number, and error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    # Return the formatted error message
    return error_message

# Custom exception class for handling US VISA-related errors
class USVISAException(Exception):
    # Initialize the custom exception with an error message and error details
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Initialize the base Exception with the error message
        # Format the error message with details and store it in the instance variable
        self.error_message = message_details(error_message, error_detail=error_detail)

    # Define how the exception should be displayed as a string
    def __str__(self):
        return self.error_message
