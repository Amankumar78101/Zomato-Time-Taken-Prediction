import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from dataclasses import dataclass




class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        """This function takes in a set of input features, loads the preprocessed data and trained model from saved artifacts,
          applies the preprocessor to scale the input features, makes predictions using the loaded model, and returns the predicted values."""
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor1.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    """It takes in several input parameters related to a delivery service, such as the delivery person's age and ratings, weather conditions, 
       road traffic density, type of order and vehicle, and festival and city information.
       It has a method called get_data_as_dataframe() which returns the input data in the form of a Pandas DataFrame."""
    def __init__(self,
                 Delivery_person_Age:float,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:float,
                 Road_traffic_density:str,
                 Festival:str,
                 City:str):
        
        self.Delivery_person_Age=Delivery_person_Age
        self.Delivery_person_Ratings=Delivery_person_Ratings
        self.Weather_conditions=Weather_conditions
        self.Vehicle_condition=Vehicle_condition
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.multiple_deliveries = multiple_deliveries
        self.Road_traffic_density=Road_traffic_density
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weather_conditions':[self.Weather_conditions],
                'Vehicle_condition':[self.Vehicle_condition],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'multiple_deliveries':[self.multiple_deliveries],
                'Road_traffic_density':[self.Road_traffic_density],
                'Festival':[self.Festival],
                'City':[self.City]}
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)



