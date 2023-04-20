import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


## Intitialize the Data Ingetion Configuration

@dataclass
class DataIngestionconfig:
    """The DataIngestionconfig class is decorated with @dataclass and has three attributes train_data_path, 
       test_data_path, and raw_data_path that have default values pointing to file paths in the artifacts directory."""
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

## create a class for Data Ingestion
class DataIngestion:
    """The DataIngestion class has an initiate_data_ingestion method that reads data from a CSV file,
       saves it to a specified file path, splits it into train and test sets, and saves them to separate 
       file paths; logs messages using the logging module; and catches and raises exceptions using the CustomException class."""
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion methods Starts')
        try:
            # Read the data using the pandas
            df=pd.read_csv(os.path.join('notebooks/data','finalTrain.csv'))
            logging.info('Dataset read as pandas Dataframe')

            # Seving the Rae data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train and test and split the data')
            # Split the train and test data
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            # Seving the train and test data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
  
            
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)






