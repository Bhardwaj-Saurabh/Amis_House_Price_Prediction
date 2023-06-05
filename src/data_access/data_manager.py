
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException


class HousingData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self, data_path: str):
        try:
            self.data_path = data_path

        except Exception as e:
            raise CustomException(e, sys)

    def read_data(self) -> pd.DataFrame:
        try:
            """
            read dataframe from given path:
            return pd.DataFrame
            """
            logging.info("Start Reading dataset")
            df = pd.read_csv(self.data_path)
            df["MSSubClass"] = df["MSSubClass"].astype("O")
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)