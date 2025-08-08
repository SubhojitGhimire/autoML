import os
import ray
import math
import psutil
import datetime
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

class RegressionInRay:
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
    
    def metric_calculate(self, y_test, y_pred):
        y_mean = np.mean(np.abs(y_test)) if np.mean(np.abs(y_test)) != 0 else 1e-8
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        rel_rmse = rmse / y_mean * 100
        log_penalty = 1 / (1 + np.log1p(rel_rmse))
        r2 = r2_score(y_test, y_pred)
        r2_scaled = (r2 + 1) / 2
        print("SCORE: ", log_penalty * r2_scaled)
        return log_penalty * r2_scaled

    def call_single_value_predictor(self):
        class ConstantValuePredictor:
            def __init__(self):
                self.constant_value = None
            def fit(self, X_train, y_train):
                self.constant_value = np.mean(y_train)
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
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            
            return resultDF, {"Score %": 100}
        except Exception as e:
            print(f"Error occurred while calling {self.algorithm_name} (Single Value Predictor) model: {e}")
            return pd.DataFrame(), {}

    def call_xgbregressor_model(self):
        def hypertune_xgbregressor_model(sample_space, self):
            try:
                parameters = {}
                parameters["n_estimators"], parameters["max_depth"], parameters["learning_rate"], parameters["subsample"], parameters["colsample_bytree"], parameters["gamma"] = sample_space
                model = XGBRegressor(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -self.metric_calculate(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning XGB Regressor model: {e}")
                return 9999
        
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_xgbregressor_model,
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
                except:
                    pass
            model = XGBRegressor(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Score %": round(self.metric_calculate(self.testData[self.label], predictionResult)*100, 2)}

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling XGBoost Regressor model: {e}")
            return pd.DataFrame(), {}
    
    def call_linearregression_model(self):
        def hypertune_linearregression_model(sample_space, self):
            try:
                parameters = {}
                parameters["fit_intercept"], parameters["normalize"], parameters["copy_X"], parameters["n_jobs"] = sample_space
                model = LinearRegression(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -self.metric_calculate(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning Linear Regression model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_linearregression_model,
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
                except:
                    pass
            model = LinearRegression(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Score %": round(self.metric_calculate(self.testData[self.label], predictionResult)*100, 2)}

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling Linear Regression model: {e}")
            return pd.DataFrame(), {}
        
    def call_randomforestregressor_model(self):
        def hypertune_randomforestregressor_model(sample_space, self):
            try:
                parameters = {}
                parameters["n_estimators"], parameters["criterion"], parameters["max_depth"], parameters["min_samples_split"], parameters["min_samples_leaf"], parameters["max_features"] = sample_space
                model = RandomForestRegressor(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -self.metric_calculate(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning Random Forest Regressor model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_randomforestregressor_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["n_estimators"], bestParameters["criterion"], bestParameters["max_depth"], bestParameters["min_samples_split"], bestParameters["min_samples_leaf"], bestParameters["max_features"] = analysis.x
                except:
                    pass
            model = RandomForestRegressor(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Score %": round(self.metric_calculate(self.testData[self.label], predictionResult)*100, 2)}

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling Random Forest Regressor model: {e}")
            return pd.DataFrame(), {}
    
    def call_svr_model(self):
        def hypertune_svr_model(sample_space, self):
            try:
                parameters = {}
                parameters["kernel"], parameters["degree"], parameters["coef0"], parameters["tol"], parameters["C"], parameters["epsilon"], parameters["shrinking"], parameters["cache_size"], parameters["max_iter"] = sample_space
                model = SVR(**parameters)
                model.fit(self.trainData[self.features], self.trainData[self.label])
                return -self.metric_calculate(self.testData[self.label], model.predict(self.testData[self.features]))
            except Exception as e:
                print(f"Error occurred while hypertuning SVR model: {e}")
                return 9999
        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_svr_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["kernel"], bestParameters["degree"], bestParameters["coef0"], bestParameters["tol"], bestParameters["C"], bestParameters["epsilon"], bestParameters["shrinking"], bestParameters["cache_size"], bestParameters["max_iter"] = analysis.x
                except:
                    pass
            model = SVR(**bestParameters)
            model.fit(self.trainData[self.features], self.trainData[self.label])
            predictionResult = model.predict(self.testData[self.features])
            predictionDF = pd.DataFrame({f"{self.label}_predicted": predictionResult}, index=self.testData.index)
            score = {"Score %": round(self.metric_calculate(self.testData[self.label], predictionResult)*100, 2)}

            fullData = pd.concat([self.trainData, self.testData])
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], predictionDF[f"{self.label}_predicted"], fullData.iloc[:, labelIdx+1:]], axis=1)
            resultDF[self.features] = self.featureEncoder.inverse_transform(resultDF[self.features])
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling SVR model: {e}")
            return pd.DataFrame(), {}
    
    def run_algorithm(self, rayDF):
        try:
            if len(rayDF) == 0:
                raise Exception("Empty DataFrame received for processing.")
            
            rayDF.reset_index(drop=True, inplace=True)
            allColumns = self.features+[self.label]+self.config["partition_columns"] if self.groupbyFlag else self.features+[self.label]
            for columnName in allColumns:
                rayDF[columnName] = rayDF[columnName].interpolate(method='linear').fillna(rayDF[columnName].bfill().ffill())

            if self.config["split_method"] == "Percentage Split":
                self.featureEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                rayDF[self.features] = self.featureEncoder.fit_transform(rayDF[self.features])

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
                    self.featureEncoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    rayDF[self.features] = self.featureEncoder.fit_transform(rayDF[self.features])
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
            elif self.algorithm_name == "XGBoost Regressor":
                result, score = self.call_xgbregressor_model()
            elif self.algorithm_name == "Linear Regression":
                result, score = self.call_linearregression_model()
            elif self.algorithm_name == "Random Forest Regressor":
                result, score = self.call_randomforestregressor_model()
            elif self.algorithm_name == "SVM Regressor":
                result, score = self.call_svr_model()
            else:
                raise Exception("Unsupported Regression algorithm specified in configuration.")   
            if result.empty:
                raise Exception("Failed to run algorithm.")
            
            return result, score
        except Exception as e:
            print(f"Error occurred while running algorithm: {e}")
            return pd.DataFrame(), {}
            
# @ray.remote
# def run_algorithm_in_ray(config, rayDFRef):
#     runAlgoInstance = AlgorithmInRay(config)
#     result, score = runAlgoInstance.run_algorithm(rayDFRef)
#     del runAlgoInstance
#     return result, score

# @ray.remote
# class GetResource:
#     def getResFunc(self):
#         cpu_percent = psutil.cpu_percent()
#         memory_percent = psutil.virtual_memory().percent
#         return (cpu_percent + memory_percent)

# class RegressionModel:
#     def __init__(self, configurationJSON, rayDFList):
#         self.configurationJSON = configurationJSON
#         self.rayDFList = rayDFList
#         self.partitionCount = len(self.rayDFList)
#         self.totalRayNodesAvailable = 1
#         self.finalResult = []
#         self.successCount = 0
#         self.failedCount = 0

#     def update_progress(self, score, force_complete=False):
#         progress_file = os.path.join(os.getcwd(), "Output", "progress.csv")
#         if os.path.exists(progress_file):
#             read_progress_df = pd.read_csv(progress_file)
#             read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Progress'] = f"{(self.successCount + self.failedCount)/self.partitionCount*100:.2f}%"
#             if force_complete:
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Progress'] = "100.00%"
#             read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'End_Time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Execution_Time'] = str(datetime.datetime.now() - self.configurationJSON["timestamp"])[:-3]
#             if self.successCount == self.partitionCount:
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = "Successfully Completed"
#             elif self.failedCount == self.partitionCount:
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = "Failed"
#             elif self.successCount + self.failedCount == self.partitionCount:
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = f"Partially Completed"
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Remarks'] = f"{self.successCount} Success, {self.failedCount} failed"
#             if score:
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Metric_Name'] = list(score.keys())[0]
#                 read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Metric_Value'] = list(score.values())[0]
#             read_progress_df.to_csv(progress_file, index=False)

#     def sort_by_best_ray_nodes(self):
#         try:
#             rayNodeList = ray.nodes()
#             nodeDict = {}
#             for node in rayNodeList:
#                 if node["Alive"] and node["Resources"]["CPU"] > 0:
#                     nodeIP = node["NodeManagerAddress"].split(":")[0]
#                     nodeActor = GetResource.options(resources={f"node:{nodeIP}":0.001}).remote()
#                     resFunc_future = nodeActor.getResFunc.remote()
#                     ready, _ = ray.wait([resFunc_future], timeout=15)
#                     if ready:
#                         nodeResourceUsage = ray.get(resFunc_future)
#                     else:
#                         nodeResourceUsage = 100 # 100% usage. No resource free in the worker node
#                     ray.kill(nodeActor)
#                     nodeDict[nodeIP] = nodeResourceUsage
#             sortedNodeList = dict(sorted(nodeDict.items(), key=lambda item: item[1]))
#             bestNodeList = [nodeKey for nodeKey in sortedNodeList.keys() if sortedNodeList[nodeKey] < 100]
#             return bestNodeList
#         except Exception as e:
#             print(f"Exception Occurred while sorting ray nodes. \nError Reads: {e}")
#             return []

#     def execute(self):
#         print(f"\n\n\n\n\n\n\n---x---\n\n\n\n\n\n\n::: NOW EXECUTING :::")
#         rayNodeList = self.sort_by_best_ray_nodes()
#         rayNodeList = [False] if not rayNodeList else rayNodeList

#         resources = {}
#         resourcesList = []
#         if (rayNodeList) and (rayNodeList[0] != False):
#             for node in rayNodeList:
#                 for nodes in ray.nodes():
#                     if node in nodes["NodeManagerAddress"] and nodes["Alive"] is True:
#                         use_resource = (nodes["Resources"]["CPU"])
#                         use_resource = round(config_instance.rayResourcePerAlgo/use_resource, 3)
#                         resources[f"node:{node}"] = use_resource
#                         resourcesList.append(resources)
#         if not resourcesList:
#             resourcesList = [{}]
#         self.totalRayNodesAvailable = len(resourcesList)

#         batchSize = config_instance.batchSize
#         if batchSize > self.partitionCount:
#             batchSize = self.partitionCount
        
#         batchIDs = []
#         batchIndex = 0
#         for rayDFRef in self.rayDFList[:batchSize]:
#             batchIDs.append(run_algorithm_in_ray.options(resources=resourcesList[batchIndex%self.totalRayNodesAvailable]).remote(self.configurationJSON, rayDFRef))
#             batchIndex += 1
        
#         while len(batchIDs):
#             doneID, batchIDs = ray.wait(batchIDs)
#             completedTaskID = doneID[0]
#             try:
#                 if self.partitionCount > batchIndex:
#                     batchIDs.append(run_algorithm_in_ray.options(resources=resourcesList[batchIndex%self.totalRayNodesAvailable]).remote(self.configurationJSON, self.rayDFList[batchIndex]))
#                     batchIndex += 1
#             except Exception as e:
#                 print(f"Exception Occurred while executing forecasting model. \nError Reads: {e}")
#                 continue

#             result, score = ray.get(completedTaskID)
#             if not result.empty:
#                 self.finalResult.append(result)
#                 self.successCount += 1
#             else:
#                 self.failedCount += 1
#             self.update_progress(score=score)

#         self.finalResult = pd.concat(self.finalResult)
#         if not self.finalResult.empty:
#             self.finalResult.reset_index(drop=True, inplace=True)
#             self.finalResult.to_csv(os.path.join(os.getcwd(), "Output", f"{self.configurationJSON['model_id']}.csv"), index=False)
#         self.update_progress(score=score, force_complete=True)
        
#         print(f"---X--- {self.configurationJSON['model_id']} COMPLETED ---X---")
            
