import os
import urllib.request as request
from src.datascience import logger
import zipfile
from src.datascience.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config

    #Train test split
    def train_test_splitting(self):
        data=pd.read_csv(self.config.data_path)

        #Splitting the data into train and test
        train,test=train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)

        logger.info("Splitted the data into train and test")
        logger.info(train.shape)
        logger.info(test.shape)
      