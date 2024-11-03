
import sys
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.entity.s3_estimator import USvisaEstimator
from us_visa_project.entity.config_entity import ModelPusherConfig
from us_visa_project.cloud_storage.aws_storage import SimpleStorageService
from us_visa_project.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initialize ModelPusher with evaluation artifacts and configuration.

        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiate the process of pushing the model to the specified S3 bucket.

        Returns:
            ModelPusherArtifact: Artifact containing details about the pushed model.
        
        Raises:
            USVISAException: Custom exception in case of failure.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading model to S3 bucket")
            
            # Save the model to S3 using the path from the evaluation artifact
            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            # Create an artifact with the details of the pushed model
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Model successfully uploaded to S3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            
            return model_pusher_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e
