from distutils import dir_util
from src.data_access.data_manager import HousingData
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                        data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise  CustomException(e,sys)
        
    def validate_inputs(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Check model inputs for na values and filter."""
        try:
            validated_data = input_data.copy()
            validated_data = validated_data[self._schema_config['features']+[TARGET_COLUMN]]
            new_vars_with_na = [
                var
                for var in self._schema_config['features']
                if var
                not in self._schema_config['categorical_vars_with_na_frequent']
                + self._schema_config['categorical_vars_with_na_missing']
                + self._schema_config['numerical_vars_with_na']
                and validated_data[var].isnull().sum() > 0
            ]
            validated_data.dropna(subset=new_vars_with_na, inplace=True)
            return validated_data
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["HouseDataInputSchema"].keys())
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report ={}
            for column in base_df.columns:
                if base_df[column].dtype != 'object':
                    d1 = base_df[column]
                    d2  = current_df[column]
                    is_same_dist = ks_2samp(d1,d2)
                    if threshold<=is_same_dist.pvalue:
                        is_found=False
                    else:
                        is_found = True 
                        status=False

                    report.update({column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                        }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report,)
            return status
        except Exception as e:
            raise CustomException(e,sys)
   

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            #Reading data from train and test file location
            train_dataframe = HousingData(train_file_path).read_data()
            test_dataframe = HousingData(test_file_path).read_data()

            #Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Test dataframe does not contain all columns.\n"
        
            if len(error_message)>0:
                raise Exception(error_message)
            

            train_dataframe = self.validate_inputs(train_dataframe)
            test_dataframe = self.validate_inputs(test_dataframe)

            #Let check data drift
            status = self.detect_dataset_drift(base_df=train_dataframe,
                                               current_df=test_dataframe)

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting Valid train and test file path.")

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)