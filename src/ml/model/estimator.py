from src.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

#Write a code to train model and check the accuracy.
class HousePriceModel:

    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e
    
