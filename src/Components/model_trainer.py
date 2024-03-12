import os
import sys
from datetime import datetime
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join(os.getcwd(),'artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models= {
                "RandomForest":RandomForestRegressor()
                ,
                "DecisionTree":DecisionTreeRegressor()
                ,
                "GradientBoosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "Linear_Regression":LinearRegression(),
                "KNeighbors":KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "CatBoost":CatBoostRegressor(logging_level='Silent')
            }

            params={
                "Linear_Regression":{},
                "RandomForest":{"n_estimators":[50,10]
                                ,
                                "max_depth":[3,6],
                                "min_samples_leaf":[1,2,3]
                                }
                                ,
                "DecisionTree":{"max_depth":[4,7],
                                "min_samples_split":[2,3]
                                },
                "GradientBoosting":{"n_estimators":[50,100],
                                    "min_samples_leaf":[1,5]
                                    },
                "AdaBoost":{"n_estimators":[50,55],
                            "learning_rate":[1.0,1.5]
                            },
                "KNeighbors":{"n_neighbors":[5,7]},
                "XGB":{"learning_rate":[0.1,0.2], "n_estimators":[90,100]},
                "CatBoost":{}
                }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("no best model found")

            logging.info("Fond the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                        )

            # predicted=best_model.predict(x_test)
            # r2_square=r2_score(y_test,predicted)
            print("best model: ")
            print(best_model,best_model_score)

            print("all models: ")
            for i,j in model_report.items():
                print(i,j)

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            # return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        

    
