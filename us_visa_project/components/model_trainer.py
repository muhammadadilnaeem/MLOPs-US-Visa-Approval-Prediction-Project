import sys
import numpy as np
import pandas as pd
from typing import Tuple
from pandas import DataFrame
from neuro_mf import ModelFactory
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Importing custom modules for logging, error handling, and configuration
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.entity.target_estimator import USvisaModel
from us_visa_project.entity.config_entity import ModelTrainerConfig
from us_visa_project.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from us_visa_project.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer with data transformation artifacts and training configuration.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Uses the ModelFactory to retrieve the best model and its performance report based on training data.
        
        :param train: Training data array with features and labels
        :param test: Test data array with features and labels
        :return: Tuple containing the best model details and a metric artifact object
        """
        try:
            logging.info("Using neuro_mf to find the best model object and generate its report")
            
            # Initialize ModelFactory with configuration path for model selection
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            # Split training and test arrays into features and labels
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            # Get the best model based on training data and expected accuracy
            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_accuracy
            )
            model_obj = best_model_detail.best_model

            # Predict on the test set and calculate performance metrics
            y_pred = model_obj.predict(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Create a ClassificationMetricArtifact to store metric scores
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise USVISAException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process by loading transformed data, training the model, 
        and saving the trained model and its associated metadata.
        
        :return: A ModelTrainerArtifact containing paths to the saved model and metrics
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        
        try:
            # Load transformed training and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            # Retrieve the best model and performance metrics
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            # Load preprocessing object to be used in the final model pipeline
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Ensure the best model's score meets the expected accuracy threshold
            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with a score higher than the base score")
                raise Exception("No best model found with score more than base score")

            # Create the final model with preprocessing and best model components
            usvisa_model = USvisaModel(preprocessing_object=preprocessing_obj, trained_model_object=best_model_detail.best_model)
            logging.info("Constructed USvisa model object with preprocessor and model")
            
            # Save the final model to the designated file path
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)

            # Create a ModelTrainerArtifact with paths to saved model and metrics
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e
