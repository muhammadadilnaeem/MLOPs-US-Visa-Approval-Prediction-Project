import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.entity.s3_estimator import USvisaEstimator
from us_visa_project.entity.target_estimator import USvisaModel
from us_visa_project.constants import TARGET_COLUMN, CURRENT_YEAR
from us_visa_project.entity.target_estimator import TargetValueMapping
from us_visa_project.entity.config_entity import ModelEvaluationConfig
from us_visa_project.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact

@dataclass
class EvaluateModelResponse:
    """
    Data class for storing model evaluation results.
    """
    trained_model_f1_score: float  # F1 score of the newly trained model
    best_model_f1_score: float  # F1 score of the best model in production (if available)
    is_model_accepted: bool  # Whether the trained model is accepted based on evaluation
    difference: float  # Difference in F1 score between the trained and production models

class ModelEvaluation:
    """
    This class handles the evaluation of a trained model against the current production model.
    """

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initialize ModelEvaluation with configuration, data ingestion artifacts, and trained model artifacts.
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Retrieve the best model currently in production from S3 storage.
        
        Returns:
            USvisaEstimator object if available, otherwise None.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name, model_path=model_path)

            if usvisa_estimator.is_model_present(model_path=model_path):
                return usvisa_estimator
            return None
        except Exception as e:
            raise USVISAException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the trained model by comparing its F1 score to the production model's F1 score.
        
        Returns:
            EvaluateModelResponse containing evaluation results.
        """
        try:
            # Load the test data for evaluation
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            test_df['company_age'] = CURRENT_YEAR - test_df['yr_of_estab']  # Add company age feature

            # Separate features and target variable
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())  # Map target values if necessary

            # Get F1 score of the trained model from its metrics
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()  # Load the best model in production if available
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)  # Predict with production model
                best_model_f1_score = f1_score(y, y_hat_best_model)  # Calculate F1 score for production model
            
            # Determine if the trained model outperforms the production model
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise USVISAException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Execute the complete model evaluation process and return evaluation artifacts.
        
        Returns:
            ModelEvaluationArtifact containing evaluation results.
        """
        try:
            evaluate_model_response = self.evaluate_model()  # Perform model evaluation
            s3_model_path = self.model_eval_config.s3_model_key_path

            # Prepare the model evaluation artifact based on evaluation results
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USVISAException(e, sys) from e
