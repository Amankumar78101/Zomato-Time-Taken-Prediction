import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # seving the preprocessor file
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor1.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be categorical-numerical and which should be scaled
            categorical_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order','Type_of_vehicle', 'Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries']
            

            # Define the custom ranking for each categorical variable
            weather_ranking = ['nan', 'Sunny', 'Cloudy', 'Windy', 'Sandstorms', 'Stormy', 'Fog']
            Road_traffic_densitys_ranking=['nan','Low','Medium','High','Jam']
            Type_of_orders_ranking=['Snack','Buffet', 'Meal', 'Drinks']
            type_of_vehicle_ranking = ['electric_scooter', 'motorcycle', 'scooter', 'bicycle']
            festival_ranking = ['nan', 'No','Yes' ]
            city_ranking = ['nan','Semi-Urban','Urban','Metropolitian' ]
            
            logging.info('Pipeline Initiated')

            ## Creating the Numerical Pipeline
            num_pipeline=Pipeline(steps=[
                                         ('imputer',SimpleImputer(strategy='median')),
                                         ('scaler',StandardScaler())])

            # Creating the Categorigal Pipeline
            cat_pipeline=Pipeline(steps=[
                                         ('imputer',SimpleImputer(strategy='most_frequent')),
                                         ('ordinalencoder',OrdinalEncoder(categories=[weather_ranking,Road_traffic_densitys_ranking,
                                                                                        Type_of_orders_ranking,type_of_vehicle_ranking,
                                                                                        festival_ranking,city_ranking])),
                                        ('scaler',StandardScaler())
                                        ])
            
            # Creating the preprocessor
            preprocessor=ColumnTransformer([
                                            ('num_pipeline',num_pipeline,numerical_cols),
                                            ('cat_pipeline',cat_pipeline,categorical_cols)
                                            ])

            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            # making a object get_data_transformation_object()
            preprocessing_obj = self.get_data_transformation_object()

            # Selecting the data to target for prediction.
            target_column_name = 'Time_taken (min)'
            # Selecting the data to drop. This data is not correlated with what I want to predict
            drop_columns = [target_column_name,'ID','Delivery_person_ID','Order_Date','Time_Orderd','Time_Order_picked',"Restaurant_latitude",'Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude']

            # This code is preparing the training data by selecting the input features and target feature from the train Dataframe. 
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            # This code is preparing the testing data by selecting the input features and target feature from the test Dataframe.
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            """This code combines the input feature array and target feature array horizontally using NumPy's np.c_ method,
              creating the final training and testing arrays for the machine learning model."""
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)
        








