
from dataclasses import dataclass

# Class representing the data ingestion artifacts
@dataclass
class DataIngestionArtifact:
    # Path to the training data file
    trained_file_path: str
    # Path to the testing data file
    test_file_path: str

# Class representing the data Validation artifacts
@dataclass
class DataValidationArtifact:

    # Status of data validation
    validation_status:bool
    # Message for data validation
    message: str
    # Path to the drift report
    drift_report_file_path: str

# Class representing the data Transformation artifacts
@dataclass
class DataTransformationArtifact:

    # Path to the transformed object file
    transformed_object_file_path:str 
    # Path to the transformed train file
    transformed_train_file_path:str
    # Path to the transformed test file
    transformed_test_file_path:str

# Class representing the model trainer artifacts
@dataclass
class ClassificationMetricArtifact:
    # Accuracy of the model
    f1_score:float
    # Precision of the model
    precision_score:float
    # Recall of the model
    recall_score:float

# Class representing the model trainer artifacts
@dataclass
class ModelTrainerArtifact:
    # Path to the trained model
    trained_model_file_path:str 
    # Metric artifact
    metric_artifact:ClassificationMetricArtifact


# Class representing the model evaluation artifacts
@dataclass
class ModelEvaluationArtifact:
    # Status of model evaluation
    is_model_accepted:bool
    # Metric artifact to compare model performance
    changed_accuracy:float
    # Path to the trained model for deployment
    s3_model_path:str 
    # Path to the trained model for deployment
    trained_model_path:str


# Class representing the model pusher artifacts
@dataclass
class ModelPusherArtifact:
    # amazon s3 bucket name
    bucket_name:str
    # Path to the trained model for deployment
    s3_model_path:str