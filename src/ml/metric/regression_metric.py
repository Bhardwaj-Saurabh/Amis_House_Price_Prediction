from src.entity.artifact_entity import RegressionMetricArtifact
from src.exception import CustomException
from sklearn.metrics import mean_squared_error, r2_score
import os,sys

def get_regression_score(y_true, y_pred)->RegressionMetricArtifact:
    try:
        mean_squared_error_score = mean_squared_error(y_true, y_pred)
        mean_root_squared_error_score = mean_squared_error(y_true, y_pred, squared=False)
        model_r2_score=r2_score(y_true, y_pred)

        regression_metric =  RegressionMetricArtifact(mean_squared_error_score=mean_squared_error_score,
                    mean_root_squared_error_score=mean_root_squared_error_score, 
                    model_r2_score=model_r2_score)
        return regression_metric
    except Exception as e:
        raise CustomException(e,sys)