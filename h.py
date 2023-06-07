path = "artifact/06_07_2023_12_21_28/data_validation/validated/train_data.csv"

import pandas as pd

df = pd.read_csv(path)
print(df.head())

from sklearn.pipeline import Pipeline


from src.constant.training_pipeline import TARGET_COLUMN

from src.utils.main_utils import save_numpy_array_data, save_object
from src.data_access.data_manager import HousingData
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils import read_yaml_file, write_yaml_file

from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler

from src.ml.preprocess import preprocessing as pp


schema_config = read_yaml_file(SCHEMA_FILE_PATH)


preprocessor = Pipeline(
[
    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    (
        "missing_imputation",
        CategoricalImputer(
            imputation_method="missing",
            variables=schema_config["categorical_vars_with_na_missing"]
        ),
    ),
    (
        "frequent_imputation",
        CategoricalImputer(
            imputation_method="frequent",
            variables=schema_config["categorical_vars_with_na_frequent"]
        ),
    ),
    # add missing indicator
    (
        "missing_indicator",
        AddMissingIndicator(variables=schema_config["numerical_vars_with_na"])
    ),
    # impute numerical variables with the mean
    (
        "mean_imputation",
        MeanMedianImputer(
            imputation_method="mean",
            variables=schema_config["numerical_vars_with_na"],
        ),
    ),
    # == TEMPORAL VARIABLES ====
    (
        "elapsed_time",
        pp.TemporalVariableTransformer(
            variables=schema_config["temporal_vars"],
            reference_variable=schema_config["ref_var"],
        ),
    ),
    ("drop_features", DropFeatures(features_to_drop=[schema_config["ref_var"]])),
    # ==== VARIABLE TRANSFORMATION =====
    ("log", LogTransformer(variables=schema_config["numericals_log_vars"])),
    (
        "binarizer",
        SklearnTransformerWrapper(
            transformer=Binarizer(threshold=0),
            variables=schema_config["binarize_vars"],
        ),
    ),
    # === mappers ===
    (
        "mapper_qual",
        pp.Mapper(
            variables=schema_config["qual_vars"],
            mappings=schema_config["qual_mappings"],
        ),
    ),
    (
        "mapper_exposure",
        pp.Mapper(
            variables=schema_config["exposure_vars"],
            mappings=schema_config["exposure_mappings"],
        ),
    ),
    (
        "mapper_finish",
        pp.Mapper(
            variables=schema_config["finish_vars"],
            mappings=schema_config["finish_mappings"],
        ),
    ),
    (
        "mapper_garage",
        pp.Mapper(
            variables=schema_config["garage_vars"],
            mappings=schema_config["garage_mappings"],
        ),
    ),
    # == CATEGORICAL ENCODING
    (
        "rare_label_encoder",
        RareLabelEncoder(
            tol=0.01, n_categories=1, variables=schema_config["categorical_vars"]
        ),
    ),
    # encode categorical variables using the target mean
    (
        "categorical_encoder",
        OrdinalEncoder(
            encoding_method="ordered",
            variables=schema_config["categorical_vars"],
        ),
    ),
    ("scaler", MinMaxScaler()),
]
)

#training dataframe
train_df = df.drop(columns=[TARGET_COLUMN], axis=1)
train_df['MSSubClass'] = train_df['MSSubClass'].astype('O')
test_df = df[TARGET_COLUMN]
print(train_df.isna().sum())

#preprocessor.fit(train_df, test_df)
