
from dataclasses import dataclass

# Class representing the data ingestion artifacts

@dataclass
class DataIngestionArtifact:
    # Path to the training data file
    trained_file_path: str
    # Path to the testing data file
    test_file_path: str


@dataclass
class DataValidationArtifact:

    # Status of data validation
    validation_status:bool
    # Message for data validation
    message: str
    # Path to the drift report
    drift_report_file_path: str