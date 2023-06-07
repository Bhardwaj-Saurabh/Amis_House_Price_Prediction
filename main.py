
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline import training_pipeline
from src.pipeline.training_pipeline import TrainingPipeline


def main():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)


if __name__ == "__main__":
    main()
