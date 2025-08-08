import os
import ray
import math
import psutil
import datetime
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

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
        metric = res.fun
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_params = res.x
        if metric > self.threshold:
            return True
        return False

class ClassificationInRay:
    def __init__(self, config):
        self.config = config

        self.algorithm_name = self.config["algorithm"]
        self.groupbyFlag = True if self.config["partition_columns"] else False
        self.label = self.config["label_column"]
        self.features = self.config["feature_columns"]
        self.tuneType = self.config["hyperparameter_tuning"]
        self.parameters = self.config["parameters"]
        self.trainData = None
        self.testData = None
        self.featureEncoder = None
        self.labelEncoder = None
    
    def call_single_value_predictor(self):
        class ConstantValuePredictor:
            def __init__(self):
                self.constant_value = None
            def fit(self, X_train, y_train):
                self.constant_value = y_train.mode()[0]
            def predict(self, X_test):
                return np.full((X_test.shape[0], ), self.constant_value)
        try:
            cvpModel = ConstantValuePredictor()
            cvpModel.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = cvpModel.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            resultDF[self.label] = self.labelEncoder.inverse_transform(resultDF[self.label])
            non_nan_mask = resultDF[f"{self.label}_predicted"].notna()
            resultDF.loc[non_nan_mask, f"{self.label}_predicted"] = self.labelEncoder.inverse_transform(resultDF.loc[non_nan_mask, f"{self.label}_predicted"].astype(int))

            return resultDF, {"Accuracy %": 100}
        except Exception as e:
            print(f"Error occurred while calling {self.algorithm_name} (Single Value Predictor) model: {e}")
            return pd.DataFrame(), {}
    
    def call_xgbclassifier_model(self):
        def hypertune_xgbclassifier_model(sample_space, self):
            try:
                parameters = {}
                parameters["n_estimators"], parameters["max_depth"], parameters["learning_rate"], parameters["subsample"], parameters["colsample_bytree"], parameters["gamma"] = sample_space
                model = XGBClassifier(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                print("Accuracy: ", accuracy_score(self.testData[self.label], model.predict(self.testData[self.features])))
                return -accuracy_score(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning XGB Classifier model: {e}")
                return 9999
        
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_xgbclassifier_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["n_estimators"], bestParameters["max_depth"], bestParameters["learning_rate"], bestParameters["subsample"], bestParameters["colsample_bytree"], bestParameters["gamma"] = analysis.x
                    print("BEST ACCURACY: ", analysis.fun)
                except:
                    pass
            model = XGBClassifier(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Accuracy %": accuracy_score(self.testData[self.label], predictionResult)*100}
            print("SCORE", score)

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            resultDF[self.label] = self.labelEncoder.inverse_transform(resultDF[self.label])
            non_nan_mask = resultDF[f"{self.label}_predicted"].notna()
            resultDF.loc[non_nan_mask, f"{self.label}_predicted"] = self.labelEncoder.inverse_transform(resultDF.loc[non_nan_mask, f"{self.label}_predicted"].astype(int))

            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling XGBoost Classifier model: {e}")
            return pd.DataFrame(), {}
    
    def call_logisticregression_model(self):
        def hypertune_logisticregression_model(sample_space, self):
            try:
                parameters = {}
                parameters["penalty"], parameters["solver"], parameters["C"], parameters["tol"], parameters["fit_intercept"], parameters["intercept_scaling"], parameters["class_weight"], parameters["max_iter"], parameters["multi_class"], parameters["l1_ratio"] = sample_space
                model = LogisticRegression(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -accuracy_score(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning Logistic Regression model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_logisticregression_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["penalty"], bestParameters["solver"], bestParameters["C"], bestParameters["tol"], bestParameters["fit_intercept"], bestParameters["intercept_scaling"], bestParameters["class_weight"], bestParameters["max_iter"], bestParameters["multi_class"], bestParameters["l1_ratio"] = analysis.x
                except:
                    pass
            model = LogisticRegression(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Accuracy %": accuracy_score(self.testData[self.label], predictionResult)*100}

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            resultDF[self.label] = self.labelEncoder.inverse_transform(resultDF[self.label])
            non_nan_mask = resultDF[f"{self.label}_predicted"].notna()
            resultDF.loc[non_nan_mask, f"{self.label}_predicted"] = self.labelEncoder.inverse_transform(resultDF.loc[non_nan_mask, f"{self.label}_predicted"].astype(int))

            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling Logistic Regression model: {e}")
            return pd.DataFrame(), {}
    
    def call_linearsvm_model(self):
        def hypertune_linearsvm_model(sample_space, self):
            try:
                parameters = {}
                parameters["penalty"], parameters["tol"], parameters["C"], parameters["fit_intercept"], parameters["intercept_scaling"], parameters["class_weight"], parameters["max_iter"] = sample_space
                model = LinearSVC(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -accuracy_score(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning LinearSVC model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_linearsvm_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimnesions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["penalty"], bestParameters["tol"], bestParameters["C"], bestParameters["fit_intercept"], bestParameters["intercept_scaling"], bestParameters["class_weight"], bestParameters["max_iter"] = analysis.x
                except:
                    pass
            model = LinearSVC(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Accuracy %": accuracy_score(self.testData[self.label], predictionResult)*100}
        
            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            resultDF[self.label] = self.labelEncoder.inverse_transform(resultDF[self.label])
            non_nan_mask = resultDF[f"{self.label}_predicted"].notna()
            resultDF.loc[non_nan_mask, f"{self.label}_predicted"] = self.labelEncoder.inverse_transform(resultDF.loc[non_nan_mask, f"{self.label}_predicted"].astype(int))
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling LinearSVC model: {e}")
            return pd.DataFrame(), {}
    
    def run_algorithm(self, rayDF):
        try:
            if len(rayDF) == 0:
                raise  Exception("Empty DataFrame received for processing.")
            
            rayDF.reset_index(drop=True, inplace=True)
            allColumns = self.features+[self.label]+self.config["partition_columns"] if self.groupbyFlag else self.features+[self.label]
            for columnName in allColumns:
                rayDF[columnName] = rayDF[columnName].interpolate(method='linear').fillna(rayDF[columnName].bfill().ffill())

            self.featureEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.labelEncoder = LabelEncoder()
            if self.config["split_method"] == "Percentage Split":
                rayDF[self.features] = self.featureEncoder.fit_transform(rayDF[self.features])
                rayDF[self.label] = self.labelEncoder.fit_transform(rayDF[self.label])

                trainTestSplit = self.config["train_test_split"].split(":")
                trainSplit = float(trainTestSplit[0]) / 100
                testSplit = float(trainTestSplit[1]) / 100
                sampleSize = len(rayDF)
                trainSize = int(trainSplit * sampleSize)
                testSize = sampleSize - trainSize
                self.trainData = rayDF.iloc[:trainSize]
                self.testData = rayDF.iloc[trainSize:]
            elif self.config["split_method"] == "Time-based Split":
                self.datetimeFormat = r"%Y-%m-%d %H:%M:%S"
                if self.config["datetime_column"]:
                    rayDF["tempDatetimeColumn"] = rayDF[self.config["datetime_column"]]
                    rayDF[self.features] = self.featureEncoder.fit_transform(rayDF[self.features])
                    rayDF[self.label] = self.labelEncoder.fit_transform(rayDF[self.label])
                    if "train_start_date" in self.config and "train_end_date" in self.config and "test_start_date" in self.config and "test_end_date" in self.config:
                        trainStartDatetime = datetime.datetime.strptime(self.config["train_start_date"], self.datetimeFormat)
                        trainEndDatetime = datetime.datetime.strptime(self.config["train_end_date"], self.datetimeFormat)
                        self.trainData = rayDF[(rayDF["tempDatetimeColumn"] >= trainStartDatetime) & (rayDF["tempDatetimeColumn"] <= trainEndDatetime)]
                        testStartDatetime = datetime.datetime.strptime(self.config["test_start_date"], self.datetimeFormat)
                        testEndDatetime = datetime.datetime.strptime(self.config["test_end_date"], self.datetimeFormat)
                        self.testData = rayDF[(rayDF["tempDatetimeColumn"] >= testStartDatetime) & (rayDF["tempDatetimeColumn"] <= testEndDatetime)]
                        self.trainData.drop("tempDatetimeColumn", axis=1, inplace=True)
                        self.testData.drop("tempDatetimeColumn", axis=1, inplace=True)
                    else:
                        raise ValueError("Invalid time-based split configuration.")
                else:
                    raise ValueError("Invalid datetime column specified in configuration.")
            else:
                raise ValueError("Invalid split method specified in configuration.")

            if len(self.trainData)<4 or len(self.testData)<2:
                raise ValueError("Not enough data points for training-evaluation.")
            
            if self.trainData[self.label].nunique()==1:
                result, score = self.call_single_value_predictor()
            elif self.algorithm_name == "XGBoost Classifier":
                result, score = self.call_xgbclassifier_model()
            elif self.algorithm_name == "Logistic Regression":
                result, score = self.call_logisticregression_model()
            elif self.algorithm_name == "Linear SVM Classifier":
                result, score = self.call_linearsvm_model()
            else:
                raise Exception("Unsupported Classification algorithm specified in configuration.")   
            if result.empty:
                raise Exception("Failed to run algorithm.")
            
            return result, score
        except Exception as e:
            print(f"Error occurred while running algorithm: {e}")
            return pd.DataFrame(), {}
            

