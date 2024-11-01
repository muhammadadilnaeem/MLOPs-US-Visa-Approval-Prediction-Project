# Import necessary modules
import os            # For file and directory handling
import logging       # For logging messages to a file
from datetime import datetime  # For handling date and time
from from_root import from_root  # For getting the root directory path of the project

# Set up the log file name with the current date and time in the format MM_DD_YYYY_HH_MM_SS
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory where log files will be saved
LOG_DIR = "logs"

# Create the full path for the log file in the logs directory under the project root directory
LOG_FILE_PATH = os.path.join(from_root(), LOG_DIR, LOG_FILE)

# Create the 'logs' directory if it does not already exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the file path where logs will be stored
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the log message format
    level=logging.DEBUG,  # Set the logging level to DEBUG to capture detailed log messages
)
