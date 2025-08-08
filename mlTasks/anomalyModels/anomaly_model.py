import os
import ray
import math
import psutil
import datetime
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, EFstrType

from configurations.config import config_instance

from functools import partial
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
class EarlyStopper:
    def __init__(self, threshold):
        self.threshold = threshold
        self.best_metric = -np.inf
        self.best_params = None
    def __call__(self, res):
        metric = -res.fun
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_params = res.x
        if metric > self.threshold:
            return True
        return False

class AnomalyInRay:
    def __init__(self, config):
        self.config = config

        self.algorithm_name = self.config["algorithm"]
        self.groupbyFlag = True if self.config["partition_columns"] else False
        self.label = self.config["label_column"]
        self.features = self.config["feature_columns"]
        self.tuneType = self.config["hyperparameter_tuning"]
        self.parameters = self.config["parameters"]
        self.fullData = None
        self.featureEncoder = None

    def call_lof_model(self):
        def hypertune_lof_model(sample_space, self):
            try:
                parameters = {}
                parameters["n_neighbors"], parameters["algorithm"], parameters["leaf_size"], parameters["metric"], parameters["p"], parameters["contamination"] = sample_space
                lof = LocalOutlierFactor(**parameters)
                outlier_labels = lof.fit_predict(self.fullData[self.features+[self.label]])
                print("OUTLIER LABELS", outlier_labels)
                outliers = outlier_labels==-1
                num_outliers = outliers.sum()
                print("NUM OUTLIERS: ", num_outliers)
                return -1-(num_outliers / len(outlier_labels))
            except Exception as e:
                print(f"Error occurred while hypertuning LOF model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_lof_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=0.95)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["n_neighbors"], bestParameters["algorithm"], bestParameters["leaf_size"], bestParameters["metric"], bestParameters["p"], bestParameters["contamination"] = analysis.x
                except:
                    pass
            lof = LocalOutlierFactor(**bestParameters)
            outlier_labels = lof.fit_predict(self.fullData[self.features+[self.label]])
            outliers = outlier_labels==-1
            num_outliers = outliers.sum()
            score = {"Outlier Ratio": round(num_outliers / len(outlier_labels), 3)}
            self.fullData["outlier"] = outliers
            self.fullData[self.features+[self.label]] = self.featureEncoder.inverse_transform(self.fullData[self.features+[self.label]])
            return self.fullData, score
        except Exception as e:
            print(f"Error occurred while calling Local Outlier Factor model: {e}")
            return pd.DataFrame(), {}
    
    def call_oneclassSVM_model(self):
        def hypertune_oneclassSVM_model(sample_space, self):
            try:
                parameters = {}
                parameters["kernel"], parameters["gamma"], parameters["degree"], parameters["coef0"], parameters["tol"], parameters["nu"], parameters["shrinking"], parameters["max_iter"] = sample_space
                ocsvm = OneClassSVM(**parameters)
                outlier_labels = ocsvm.fit_predict(self.fullData[self.features+[self.label]])
                num_outliers = (outlier_labels==-1).sum()
                return -1-(num_outliers / len(outlier_labels))
            except Exception as e:
                print(f"Error occurred while hypertuning One-Class SVM model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_oneclassSVM_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=0.95)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["kernel"], bestParameters["gamma"], bestParameters["degree"], bestParameters["coef0"], bestParameters["tol"], bestParameters["nu"], bestParameters["shrinking"], bestParameters["max_iter"] = analysis.x
                except:
                    pass
            ocsvm = OneClassSVM(**bestParameters)
            ocsvm.fit(self.fullData[self.features+[self.label]])
            outlier_labels = ocsvm.predict(self.fullData[self.features+[self.label]])
            outliers = outlier_labels == -1
            num_outliers = outliers.sum()
            score = {"Outlier Ratio": round(num_outliers / len(outlier_labels), 3)}
            self.fullData["outlier"] = outliers
            self.fullData[self.features+[self.label]] = self.featureEncoder.inverse_transform(self.fullData[self.features+[self.label]])
            return self.fullData, score
        except Exception as e:
            print(f"Error occurred while calling One-Class SVM model: {e}")
            return pd.DataFrame(), {}

    def call_catBoost_model(self):
        def hypertune_catBoost_model(sampleSpace, self):
            try:
                parameters = {}
                parameters["iterations"], parameters["learning_rate"], parameters["depth"], parameters["l2_leaf_reg"], parameters["random_strength"], parameters["bagging_temperature"], parameters["border_count"], parameters["grow_policy"], parameters["subsample"], parameters["colsample_bylevel"] = sampleSpace
                model = CatBoostRegressor(**parameters)
                model.fit(self.fullData[self.features], self.fullData[self.label])
                loss_values = model.get_feature_importance(type='LossFunctionChange', data=Pool(self.fullData[self.features], self.fullData[self.label]))
                predictions = model.predict(self.fullData[self.features])
                residuals = np.abs(self.fullData[self.label] - predictions)
                threshold = np.percentile(residuals, 95)
                outliers = residuals > threshold
                num_outliers = outliers.sum()
                return -1-(num_outliers / len(outliers))
            except Exception as e:
                print(f"Error occurred while hypertuning CatBoost Outlier Detection model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_catBoost_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=0.95)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["iterations"], bestParameters["learning_rate"], bestParameters["depth"], bestParameters["l2_leaf_reg"], bestParameters["random_strength"], bestParameters["bagging_temperature"], bestParameters["border_count"], bestParameters["grow_policy"], bestParameters["subsample"], bestParameters["colsample_bylevel"] = analysis.x
                except:
                    pass
            model = CatBoostRegressor(**bestParameters)
            model.fit(self.fullData[self.features], self.fullData[self.label])
            loss_values = model.get_feature_importance(type='LossFunctionChange', data=Pool(self.fullData[self.features], self.fullData[self.label]))
            predictions = model.predict(self.fullData[self.features])
            residuals = np.abs(self.fullData[self.label] - predictions)
            threshold = np.percentile(residuals, 95)
            outliers = residuals > threshold
            num_outliers = outliers.sum()
            score = {"Outlier Ratio": round(num_outliers / len(outliers), 3)}
            self.fullData["outlier"] = outliers
            self.fullData[self.features+[self.label]] = self.featureEncoder.inverse_transform(self.fullData[self.features+[self.label]])
            return self.fullData, score
        except Exception as e:
            print(f"Error occurred while calling CatBoost Outlier Detection model: {e}")
            return pd.DataFrame(), {}
    
    def run_algorithm(self, rayDF):
        try:
            if len(rayDF) == 0:
                raise Exception("Empty DataFrame received for processing.")
            
            rayDF.reset_index(drop=True, inplace=True)
            allColumns = self.features+[self.label]+self.config["partition_columns"] if self.groupbyFlag else self.features+[self.label]
            for columnName in allColumns:
                rayDF[columnName] = rayDF[columnName].interpolate(method='linear').fillna(rayDF[columnName].bfill().ffill())
            
            self.fullData = rayDF.copy()
            self.featureEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.fullData[self.features+[self.label]] = self.featureEncoder.fit_transform(self.fullData[self.features+[self.label]])

            if len(self.fullData) < 4:
                raise ValueError("Not enough data points for anomaly detection.")
            
            if self.algorithm_name == "Local Outlier Factor":
                result, score = self.call_lof_model()
            elif self.algorithm_name == "One-Class SVM":
                result, score = self.call_oneclassSVM_model()
            elif self.algorithm_name == "CatBoost Outlier Detector":
                result, score = self.call_catBoost_model()
            else:
                raise Exception("Unsupported Outlier Detection algorithm specified in configuration.")   
            if result.empty:
                raise Exception("Failed to run algorithm.")
            
            return result, score
        except Exception as e:
            print(f"Error occurred while running algorithm: {e}")
            return pd.DataFrame(), {}
