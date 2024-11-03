
import os
import sys

from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.components.data_ingestion import DataIngestion
from us_visa_project.components.data_validation import DataValidation

from us_visa_project.components.data_transformation import DataTransformation
from us_visa_project.components.model_trainer import ModelTrainer
from us_visa_project.components.model_evaluation import ModelEvaluation
from us_visa_project.components.model_pusher import ModelPusher

from us_visa_project.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
)

from us_visa_project.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact,
    ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
)


class TrainPipeline:
    """
    This class defines the pipeline to train and deploy a machine learning model 
    for US visa project, handling each stage of the process.
    """
    def __init__(self):
    #     # Initialize configuration for each component in the pipeline
          self.data_ingestion_config = DataIngestionConfig()
          self.data_validation_config = DataValidationConfig()
          self.data_transformation_config = DataTransformationConfig()
          self.model_trainer_config = ModelTrainerConfig()
          self.model_evaluation_config = ModelEvaluationConfig()
          self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts the data ingestion component to gather and prepare raw data.
        
        Output: Returns the artifact containing train and test datasets.
        """
        try:
            logging.info("Starting data ingestion process.")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")
            return data_ingestion_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Starts the data validation component to ensure data quality.
        
        Output: Returns the artifact confirming data validation results.
        """
        try:
            logging.info("Starting data validation process.")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed.")
            return data_validation_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Starts the data transformation component to prepare data for model training.
        
        Output: Returns the artifact containing transformed data.
        """
        try:
            logging.info("Starting data transformation process.")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise USVISAException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Starts the model training component to train a machine learning model.
        
        Output: Returns the artifact containing model details and metrics.
        """
        try:
            logging.info("Starting model training process.")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise USVISAException(e, sys)

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, 
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Starts the model evaluation component to assess model performance.
        
        Output: Returns the artifact containing evaluation results.
        """
        try:
            logging.info("Starting model evaluation process.")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise USVISAException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Starts the model deployment process to push the trained model to production.
        
        Output: Returns the artifact containing model deployment details.
        """
        try:
            logging.info("Starting model pushing process.")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise USVISAException(e, sys)

    def run_pipeline(self) -> None:
        """
        Executes the full training pipeline, from data ingestion to model deployment.
        
        Output: None. Runs the entire process, logging and handling each component.
        """
        try:
            logging.info("Starting the pipeline execution.")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_artifact=data_validation_artifact
            )
            logging.info("Data transformation completed successfully.")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info("Model training completed successfully.")
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted, stopping the pipeline.")
                return None
            
            self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            logging.info("Pipeline execution completed successfully.")

        except Exception as e:
            raise USVISAException(e, sys)
