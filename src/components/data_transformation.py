import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data, save_object
from src.data_access.data_manager import HousingData
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file, write_yaml_file

from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from src.ml.preprocess import preprocessing as pp


class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def get_data_transformer_object(self)->Pipeline:
        try:
            preprocessor = Pipeline(
            [
                # ===== IMPUTATION =====
                # impute categorical variables with string missing
                (
                    "missing_imputation",
                    CategoricalImputer(
                        imputation_method="missing",
                        variables=self._schema_config["categorical_vars_with_na_missing"]
                    ),
                ),
                (
                    "frequent_imputation",
                    CategoricalImputer(
                        imputation_method="frequent",
                        variables=self._schema_config["categorical_vars_with_na_frequent"]
                    ),
                ),
                # add missing indicator
                (
                    "missing_indicator",
                    AddMissingIndicator(variables=self._schema_config["numerical_vars_with_na"])
                ),
                # impute numerical variables with the mean
                (
                    "mean_imputation",
                    MeanMedianImputer(
                        imputation_method="mean",
                        variables=self._schema_config["numerical_vars_with_na"],
                    ),
                ),
                # == TEMPORAL VARIABLES ====
                (
                    "elapsed_time",
                    pp.TemporalVariableTransformer(
                        variables=self._schema_config["temporal_vars"],
                        reference_variable=self._schema_config["ref_var"],
                    ),
                ),
                ("drop_features", DropFeatures(features_to_drop=[self._schema_config["ref_var"]])),
                # ==== VARIABLE TRANSFORMATION =====
                ("log", LogTransformer(variables=self._schema_config["numericals_log_vars"])),
                (
                    "binarizer",
                    SklearnTransformerWrapper(
                        transformer=Binarizer(threshold=0),
                        variables=self._schema_config["binarize_vars"],
                    ),
                ),
                # === mappers ===
                (
                    "mapper_qual",
                    pp.Mapper(
                        variables=self._schema_config["qual_vars"],
                        mappings=self._schema_config["qual_mappings"],
                    ),
                ),
                (
                    "mapper_exposure",
                    pp.Mapper(
                        variables=self._schema_config["exposure_vars"],
                        mappings=self._schema_config["exposure_mappings"],
                    ),
                ),
                (
                    "mapper_finish",
                    pp.Mapper(
                        variables=self._schema_config["finish_vars"],
                        mappings=self._schema_config["finish_mappings"],
                    ),
                ),
                (
                    "mapper_garage",
                    pp.Mapper(
                        variables=self._schema_config["garage_vars"],
                        mappings=self._schema_config["garage_mappings"],
                    ),
                ),
                # == CATEGORICAL ENCODING
                (
                    "rare_label_encoder",
                    RareLabelEncoder(
                        tol=0.01, n_categories=1, variables=self._schema_config["categorical_vars"]
                    ),
                ),
                # encode categorical variables using the target mean
                (
                    "categorical_encoder",
                    OrdinalEncoder(
                        encoding_method="ordered",
                        variables=self._schema_config["categorical_vars"],
                    ),
                ),
                ("scaler", MinMaxScaler()),
            ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            train_df = HousingData(self.data_validation_artifact.valid_train_file_path).read_data()
            # add MSSubClass to the list of categorical variables
            train_df['MSSubClass'] = train_df['MSSubClass'].astype('O')
            test_df = HousingData(self.data_validation_artifact.valid_test_file_path).read_data()
            test_df['MSSubClass'] = train_df['MSSubClass'].astype('O')
            preprocessor = self.get_data_transformer_object()


            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_train_df['MSSubClass'] = input_feature_train_df['MSSubClass'].astype('O')
            target_feature_train_df = train_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_test_df['MSSubClass'] = input_feature_test_df['MSSubClass'].astype('O')
            target_feature_test_df = test_df[TARGET_COLUMN]

            target_feature_train_df = np.log(train_df[TARGET_COLUMN])

            preprocessor_object = preprocessor.fit(input_feature_train_df, target_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            

            train_arr = np.c_[transformed_input_train_feature, target_feature_train_df]
            test_arr = np.c_[ transformed_input_test_feature, target_feature_test_df]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
