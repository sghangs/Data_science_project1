import os
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/sghangs/Data_science_project1.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="sghangs"
os.environ["MLFLOW_TRACKING_PASSWORD"]="ac309cca80364e64927e46b428a7f3e5a0d1bdbc"

import urllib.request as request
from urllib.parse import urlparse
from src.datascience import logger
import joblib
from src.datascience.entity.config_entity import ModelEvaluationConfig
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import mlflow
from src.datascience.utils.common import save_json
import numpy as np
from pathlib import Path

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config

    #Evaluating the model
    def eval_metrics(self,actual,pred):
        rmse=np.sqrt(mean_squared_error(actual,pred))
        mae=mean_absolute_error(actual,pred)
        r2=r2_score(actual,pred)
        return rmse,mae,r2
    
    def log_into_mlflow(self):

        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        test_x=test_data.drop([self.config.target_column],axis=1)
        test_y=test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse,mae,r2) = self.eval_metrics(test_y,predicted_qualities)

            #saving metrics as local
            scores={"rmse":rmse,"mae":mae,"r2":r2}
            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            #Model registry does not work with file store
            if tracking_url_type_store!="file":
                mlflow.sklearn.log_model(model,"model",registered_model_name="ElasticNetModel")
            else:
                mlflow.sklearn.log_model(model,"model")


