
from src.entity.config_entity import TrainingPipelineConfig

import os 
import sys
from src.logger import logging
from src.exception import CustomException




class TrainingPipeline:
    is_pipeline_running = False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_validation(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_evaluation(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_pusher(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)
        
    def run_pipeline(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)