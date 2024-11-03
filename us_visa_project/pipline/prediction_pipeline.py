import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from us_visa_project.logger import logging
from us_visa_project.exception import USVISAException
from us_visa_project.utils.main_utils import read_yaml_file
from us_visa_project.entity.s3_estimator import USvisaEstimator
from us_visa_project.entity.config_entity import USvisaPredictorConfig


class USvisaData:
    """
    Class to represent the input data structure for a US visa application.
    """

    def __init__(self,
                 continent,
                 education_of_employee,
                 has_job_experience,
                 requires_job_training,
                 no_of_employees,
                 region_of_employment,
                 prevailing_wage,
                 unit_of_wage,
                 full_time_position,
                 company_age):
        """
        Initialize USvisaData with input features for prediction.
        
        :param continent: Continent of the applicant's home country
        :param education_of_employee: Education level of the employee
        :param has_job_experience: Whether the employee has job experience (True/False)
        :param requires_job_training: Whether the position requires job training (True/False)
        :param no_of_employees: Number of employees at the company
        :param region_of_employment: Employment region
        :param prevailing_wage: Wage offered to the employee
        :param unit_of_wage: Unit of wage (e.g., hourly, weekly)
        :param full_time_position: Whether the position is full-time (True/False)
        :param company_age: Age of the company in years
        """
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_usvisa_input_data_frame(self) -> DataFrame:
        """
        Convert input data into a Pandas DataFrame.
        
        :return: DataFrame with model input features
        """
        try:
            usvisa_input_dict = self.get_usvisa_data_as_dict()
            return DataFrame(usvisa_input_dict)
        except Exception as e:
            raise USVISAException(e, sys) from e

    def get_usvisa_data_as_dict(self):
        """
        Convert input data into a dictionary format.

        :return: Dictionary with model input features
        """
        logging.info("Entered get_usvisa_data_as_dict method of USvisaData class")
        
        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logging.info("Created USvisa data dictionary")
            logging.info("Exited get_usvisa_data_as_dict method of USvisaData class")
            
            return input_data
        except Exception as e:
            raise USVISAException(e, sys) from e


class USvisaClassifier:
    """
    Classifier to handle the prediction pipeline for US visa applications.
    """

    def __init__(self, prediction_pipeline_config: USvisaPredictorConfig = USvisaPredictorConfig()) -> None:
        """
        Initialize the classifier with the prediction configuration.

        :param prediction_pipeline_config: Configuration for the prediction pipeline
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise USVISAException(e, sys)

    def predict(self, dataframe) -> str:
        """
        Perform prediction using the trained model.
        
        :param dataframe: DataFrame containing input data for prediction
        :return: Prediction result as a string
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")

            # Load the trained model using specified S3 bucket and model path
            model = USvisaEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            
            # Make prediction
            result = model.predict(dataframe)
            
            return result
        except Exception as e:
            raise USVISAException(e, sys)
