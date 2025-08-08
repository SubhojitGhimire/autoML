import os
import ray
import math
import json
import psutil
import datetime
import numpy as np
import pandas as pd

import xgboost as xgb
import tensorflow as tf
from neuralprophet import NeuralProphet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

from configurations.config import config_instance
from configurations.tuning_parameters import get_tune_parameters

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


class ForecastingInRay:
    def __init__(self, config):
        self.config = config
        
        self.algorithm_name = self.config["algorithm"]
        self.datetime_column = self.config["datetime_column"]
        self.datetimeFormat = r"%Y-%m-%d %H:%M:%S"

        self.groupbyFlag = True if self.config["partition_columns"] else False
        self.label = self.config["label_column"]
        self.tuneType = self.config["hyperparameter_tuning"]
        self.parameters = self.config["parameters"]
        self.trainData = None
        self.testData = None
        self.forecastStartDatetime = None
        self.forecastEndDatetime = None
        self.timeintervalFreq = None
    
    def metric_calculate(self, y_test, y_pred):
        y_mean = np.mean(np.abs(y_test)) if np.mean(np.abs(y_test)) != 0 else 1e-8
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        rel_rmse = rmse / y_mean * 100
        log_penalty = 1 / (1 + np.log1p(rel_rmse))
        r2 = r2_score(y_test, y_pred)
        r2_scaled = (r2 + 1) / 2
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
            trainData = self.trainData.copy()
            trainData.set_index(self.datetime_column, inplace=True)
            cvpModel = ConstantValuePredictor()
            cvpModel.fit(trainData.drop([self.label], axis=1), trainData[self.label])

            testData = self.testData.copy()
            testData.set_index(self.datetime_column, inplace=True)
            fullData = pd.concat([trainData, testData])

            forecastRange = pd.date_range(start=self.forecastStartDatetime, end=self.forecastEndDatetime, freq=self.timeintervalFreq)
            forecastResult = pd.Series(cvpModel.predict(forecastRange), index=forecastRange, name=f"{self.label}_forecasted")
            forecastDF = pd.DataFrame({f"{self.label}_forecasted": forecastResult})
            
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastDF], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            
            return resultDF, {"Score %": 100}
        except Exception as e:
            print(f"Error occurred while calling {self.algorithm_name} (Single Value Predictor) model: {e}")
            return pd.DataFrame(), {}   

    def call_arima_model(self):
        print("::: STARTING ARIMA MODEL :::")
        def hypertune_arima_model(sample_space, self):
            try:
                parameters = {}
                parameters["p"], parameters["d"], parameters["q"] = sample_space
                trainData = self.trainData.copy()
                trainData.set_index(self.datetime_column, inplace=True)
                arima_order = (parameters["p"], parameters["d"], parameters["q"])
                arima_model = ARIMA(trainData[self.label], order=arima_order, freq=self.timeintervalFreq)
                arima_model = arima_model.fit()

                testData = self.testData.copy()
                testData.set_index(self.datetime_column, inplace=True)
                evalResult = arima_model.predict(start=testData.index[0], end=testData.index[-1])
                return -self.metric_calculate(testData[self.label], evalResult)
            except Exception as e:
                print(f"Error occurred while hypertuning ARIMA model: {e}")
                return 9999

        try:
            bestParameters = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_arima_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=90)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["p"], bestParameters["d"], bestParameters["q"] = analysis.x
                except:
                    pass
            if "p" not in bestParameters:   bestParameters["p"] = 5
            if "d" not in bestParameters:   bestParameters["d"] = 1
            if "q" not in bestParameters:   bestParameters["q"] = 0

            trainData = self.trainData.copy()
            trainData.set_index(self.datetime_column, inplace=True)
            arima_order = (bestParameters["p"], bestParameters["d"], bestParameters["q"])
            arima_model = ARIMA(trainData[self.label], order=arima_order, freq=self.timeintervalFreq)
            arima_model = arima_model.fit()
            testData = self.testData.copy()
            testData.set_index(self.datetime_column, inplace=True)
            evalResult = arima_model.predict(start=testData.index[0], end=testData.index[-1])
            score = {"Score %": round(self.metric_calculate(testData[self.label], evalResult)*100, 2)}
            
            fullData = pd.concat([self.trainData, self.testData])
            fullData.set_index(self.datetime_column, inplace=True)
            arima_order = (bestParameters["p"], bestParameters["d"], bestParameters["q"])
            bestModel = ARIMA(fullData[self.label], order=arima_order, freq=self.timeintervalFreq)
            bestModel = bestModel.fit()

            forecastResult = bestModel.predict(start=self.forecastStartDatetime, end=self.forecastEndDatetime)
            forecastDF = pd.DataFrame({f"{self.label}_forecasted": forecastResult})
            
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastDF], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])

            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling ARIMA model: {e}")
            return pd.DataFrame(), {}

    def call_sarima_model(self):
        def hypertune_sarima_model(sample_space, self):
            try:
                parameters = {}
                parameters["p"], parameters["d"], parameters["q"], parameters["P"], parameters["D"], parameters["Q"], parameters["s"] = sample_space
                trainData = self.trainData.copy()
                trainData.set_index(self.datetime_column, inplace=True)
                sarima_order = (parameters["p"], parameters["d"], parameters["q"])
                sarima_seasonal_order = (parameters["P"], parameters["D"], parameters["Q"], parameters["s"])
                sarima_model = SARIMAX(trainData[self.label], order=sarima_order, seasonal_order=sarima_seasonal_order)
                sarima_model = sarima_model.fit()

                testData = self.testData.copy()
                testData.set_index(self.datetime_column, inplace=True)
                evalResult = sarima_model.predict(start=testData.index[0], end=testData.index[-1])
                return -self.metric_calculate(testData[self.label], evalResult)
            except Exception as e:
                print(f"Error occurred while hypertuning SARIMA model: {e}")
                return 9999
        
        try:
            bestParameters = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_sarima_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=0.01)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["p"], bestParameters["d"], bestParameters["q"], bestParameters["P"], bestParameters["D"], bestParameters["Q"], bestParameters["s"] = analysis.x
                except:
                    pass
            if "p" not in bestParameters:   bestParameters["p"] = 5
            if "d" not in bestParameters:   bestParameters["d"] = 1
            if "q" not in bestParameters:   bestParameters["q"] = 0
            if "P" not in bestParameters:   bestParameters["P"] = 0
            if "D" not in bestParameters:   bestParameters["D"] = 0
            if "Q" not in bestParameters:   bestParameters["Q"] = 0
            if "s" not in bestParameters:   bestParameters["s"] = 1

            trainData = self.trainData.copy()
            trainData.set_index(self.datetime_column, inplace=True)
            sarima_order = (bestParameters["p"], bestParameters["d"], bestParameters["q"])
            sarima_seasonal_order = (bestParameters["P"], bestParameters["D"], bestParameters["Q"], bestParameters["s"])
            sarima_model = SARIMAX(trainData[self.label], order=sarima_order, seasonal_order=sarima_seasonal_order)
            sarima_model = sarima_model.fit()
            testData = self.testData.copy()
            testData.set_index(self.datetime_column, inplace=True)
            evalResult = sarima_model.predict(start=testData.index[0], end=testData.index[-1])
            score = {"Score %": round(self.metric_calculate(testData[self.label], evalResult)*100, 2)}
            
            fullData = pd.concat([self.trainData, self.testData])
            fullData.set_index(self.datetime_column, inplace=True)
            sarima_order = (bestParameters["p"], bestParameters["d"], bestParameters["q"])
            sarima_seasonal_order = (bestParameters["P"], bestParameters["D"], bestParameters["Q"], bestParameters["s"])
            sarima_model = SARIMAX(fullData[self.label], order=sarima_order, seasonal_order=sarima_seasonal_order)
            sarima_model = sarima_model.fit()

            forecastResult = sarima_model.predict(start=self.forecastStartDatetime, end=self.forecastEndDatetime)
            forecastDF = pd.DataFrame({f"{self.label}_forecasted": forecastResult})
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastDF], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling SARIMA model: {e}")
            return pd.DataFrame(), {}
    
    def call_neuralprophet_model(self):
        def hypertune_neuralprophet_model(sample_space, self):
            try:
                parameters = {}
                parameters["learning_rate"], parameters["yearly_seasonality"], parameters["weekly_seasonality"], parameters["daily_seasonality"], parameters["seasonality_mode"], parameters["changepoints_range"] = sample_space
                trainData = self.trainData.copy()
                trainData.rename(columns={self.label:"y", self.datetime_column:"ds"}, inplace=True)
                neuralprophet_model = NeuralProphet(
                    learning_rate=parameters["learning_rate"],
                    yearly_seasonality=parameters["yearly_seasonality"],
                    weekly_seasonality=parameters["weekly_seasonality"],
                    daily_seasonality=parameters["daily_seasonality"],
                    seasonality_mode=parameters["seasonality_mode"],
                    changepoints_range=parameters["changepoints_range"]
                )
                neuralprophet_model.fit(trainData[["ds", "y"]], freq=self.timeintervalFreq)

                testData = self.testData.copy()
                testData.rename(columns={self.label:"y", self.datetime_column:"ds"}, inplace=True)
                evalResult = neuralprophet_model.predict(testData[["ds", "y"]])
                print(f"hypertune completed ### DATA ###\n{evalResult}")
                return -self.metric_calculate(testData["y"], evalResult["yhat1"])
            except Exception as e:
                print(f"Error occurred while hypertuning Neural Prophet model: {e}")
                return 9999

        try:
            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_neuralprophet_model,
                        self = self
                    )
                    early_stopper = EarlyStopper(threshold=0.01)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["learning_rate"], bestParameters["yearly_seasonality"], bestParameters["weekly_seasonality"], bestParameters["daily_seasonality"], bestParameters["seasonality_mode"], bestParameters["changepoints_range"] = analysis.x
                    if analysis.fun != 9999:
                        score = {"Score %": round(-analysis.fun * 100, 2)}
                except:
                    pass
            
            fullData = pd.concat([self.trainData, self.testData])
            fullData.rename(columns={self.label:"y", self.datetime_column:"ds"}, inplace=True)
            neuralprophet_model = NeuralProphet(**bestParameters)
            neuralprophet_model.fit(fullData[["ds", "y"]], freq=self.timeintervalFreq)

            forecastRange = pd.date_range(start=self.forecastStartDatetime, end=self.forecastEndDatetime, freq=self.timeintervalFreq)
            forecastDF = pd.DataFrame({"ds": forecastRange})
            forecastDF["y"] = np.nan
            forecastResult = neuralprophet_model.predict(forecastDF)

            fullData.set_index("ds", inplace=True)
            forecastResult.set_index("ds", inplace=True)
            labelIdx = fullData.columns.get_loc("y")
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastResult["yhat1"]], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            resultDF.rename(columns={"y":self.label, "yhat1":f"{self.label}_forecasted"}, inplace=True)
            resultDF.reset_index(drop=False, names=self.datetime_column, inplace=True)

            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling Neural Prophet model: {e}")
            return pd.DataFrame(), {}    
    
    def call_ablstm_model(self):
        def hypertune_ablstm_model(sample_space, self, X_train, Y_train, X_test, Y_test, y_scaler, return_model=False):
            try:
                class Attention(tf.keras.layers.Layer):
                    def __init__(self, units):
                        super(Attention, self).__init__()
                        self.W = tf.keras.layers.Dense(units)
                        self.V = tf.keras.layers.Dense(1)
                    def __call__(self, inputs):
                        score = tf.nn.tanh(self.W(inputs))
                        attention_weights = tf.nn.softmax(self.V(score), axis=1)
                        context_vector = attention_weights * inputs
                        context_vector = tf.reduce_sum(context_vector, axis=1)
                        return context_vector
                
                inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
                AttentionUnit = Attention(sample_space["Attention Units"])
                x = tf.keras.layers.LSTM(sample_space["LSTM Units"], return_sequences=True)(inputs)
                x = AttentionUnit(x)
                x = tf.keras.layers.Dense(1)(x)
                model = tf.keras.Model(inputs=inputs, outputs=x)
                callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=0)
                model.compile(optimizer="adam", loss="mse")
                model.fit(X_train, Y_train, epochs=250,callbacks=[callback])
                Y_pred = model.predict(X_test)
                Y_pred = y_scaler.inverse_transform(Y_pred)
                score = -self.metric_calculate(Y_test, Y_pred)
                if return_model:
                    return model, -score
                else:
                    return score
            except Exception as e:
                print(f"Error occurred while hypertuning Attention Based LSTM model: {e}")
                return 9999
        
        try:
            n_steps = 7
            stepColumns = []
            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            forecastResult = []
            lstmFullData = pd.concat([self.trainData, self.testData])
            lstmFullData.set_index(self.datetime_column, inplace=True)
            lstmFullData[self.label] = lstmFullData[self.label].astype(float)
            for i in range(1, n_steps + 1):
                stepColumns.append(f"{self.label}_step_{i}")
                lstmFullData[f"{self.label}_step_{i}"] = lstmFullData[self.label].shift(i)
            lstmFullData.fillna((lstmFullData.mean() + lstmFullData.median()) / 2, inplace=True)
            X = lstmFullData[stepColumns]
            Y = lstmFullData[[self.label]]
            x_scaler.fit(X)
            y_scaler.fit(Y)
            
            X_train = X.loc[self.trainData[self.datetime_column].min():self.trainData[self.datetime_column].max()]
            X_train = x_scaler.transform(X_train)
            X_train = np.flip(X_train, axis=1)
            X_train = X_train.reshape(-1, n_steps, 1)
            Y_train = Y.loc[self.trainData[self.datetime_column].min():self.trainData[self.datetime_column].max()]
            Y_train = y_scaler.transform(Y_train)
            Y_train = Y_train.reshape(-1, 1, 1)

            X_test = X.loc[self.testData[self.datetime_column].min():self.testData[self.datetime_column].max()]
            X_test = x_scaler.transform(X_test)
            X_test = np.flip(X_test, axis=1)
            X_test = X_test.reshape(-1, n_steps, 1)
            Y_test = Y.loc[self.testData[self.datetime_column].min():self.testData[self.datetime_column].max()]
            
            bestParameters = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_ablstm_model,
                        self = self,
                        X_train = X_train,
                        Y_train = Y_train,
                        X_test = X_test,
                        Y_test = Y_test,
                        y_scaler = y_scaler
                    )
                    early_stopper = EarlyStopper(threshold=0.01)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["LSTM Units"], bestParameters["Attention Units"] = analysis.x
                except:
                    pass
            if "LSTM Units" not in bestParameters:  bestParameters["LSTM Units"] = 64
            if "Attention Units" not in bestParameters:  bestParameters["Attention Units"] = 64

            model, score = hypertune_ablstm_model(
                config=bestParameters,
                self=self,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                y_scaler=y_scaler,
                return_model=True
            )
            forecastRange = pd.DataFrame(pd.date_range(start=self.forecastStartDatetime, end=self.forecastEndDatetime, freq=self.timeintervalFreq), columns=[self.datetime_column])
            forecastPeriod = len(forecastRange)
            forecastDate = np.array([self.forecastStartDatetime + self.timeintervalFreq * x for x in range(0, forecastPeriod)])
            forecastDate = forecastDate.reshape(-1, 1)

            historicalData = X_test[-1].reshape(-1, n_steps,  1)
            betweenTestForecastRange = pd.DataFrame(pd.date_range(start=self.testData[self.datetime_column].max() + self.timeintervalFreq, end=self.forecastEndDatetime, freq=self.timeintervalFreq), columns=[self.datetime_column])
            betweenTestForecastPeriod = len(betweenTestForecastRange)
            for _ in range(betweenTestForecastPeriod):
                predResult = model.predict(historicalData)
                forecastResult.append(predResult[0, 0])
                historicalData = np.roll(historicalData, shift=-1)
                historicalData[0, -1, 0] = predResult
            
            forecastResult = np.array(forecastResult)
            forecastResult = forecastResult.reshape(-1, 1)
            forecastResult = y_scaler.inverse_transform(forecastResult)
            forecastResult = forecastResult[-forecastPeriod:]
            forecastDF = pd.DataFrame({self.datetime_column: forecastDate.flatten(), f"{self.label}_forecasted": forecastResult.flatten()})
            forecastDF.set_index(self.datetime_column, inplace=True)

            fullData = pd.concat([self.trainData, self.testData])
            fullData.set_index(self.datetime_column, inplace=True)
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastDF], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])

            return resultDF, {"Score %": score}
        except Exception as e:
            print(f"Error occurred while calling Attention Based LSTM model: {e}")
            return pd.DataFrame(), {}
    
    def call_xgbforecaster_model(self):
        def hypertune_xgbforecaster_model(sample_space, self, y_test, d_train, d_test):
            try:
                parameters = {}
                parameters["n_estimators"], parameters["max_depth"], parameters["learning_rate"], parameters["subsample"], parameters["colsample_bytree"], parameters["gamma"] = sample_space
                model = xgb.train(parameters, d_train, num_boost_round=parameters["n_estimators"])
                return -self.metric_calculate(y_test, model.predict(d_test))
            except Exception as e:
                print(f"Error occurred while hypertuning XGB Forecaster model: {e}")
                return 9999
        
        try:
            engineeredFeatures = [
                "hour", "minute", "second", "day_of_month", "day_of_week", "day_of_year", 
                "month_of_year", "week_of_year", "year", "quarter",
                "is_weekend", "is_month_start", "is_month_end", 
                "is_year_start", "is_year_end", "is_quarter_start", "is_quarter_end",
                "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month",
                "fourier_sin_365", "fourier_cos_365", "fourier_sin_30", "fourier_cos_30"
            ]
            def engineer_features(engineerDF, scaler=None):
                # Engineering DataTime Features
                engineerDF['hour'] = engineerDF[self.datetime_column].dt.hour
                engineerDF['minute'] = engineerDF[self.datetime_column].dt.minute
                engineerDF['second'] = engineerDF[self.datetime_column].dt.second
                engineerDF["day_of_month"] = engineerDF[self.datetime_column].dt.day
                engineerDF["day_of_week"] = engineerDF[self.datetime_column].dt.weekday
                engineerDF["day_of_year"] = engineerDF[self.datetime_column].dt.dayofyear
                engineerDF["month_of_year"] = engineerDF[self.datetime_column].dt.month
                engineerDF["week_of_year"] = engineerDF[self.datetime_column].dt.isocalendar().week
                engineerDF['year'] = engineerDF[self.datetime_column].dt.year
                engineerDF["quarter"] = engineerDF[self.datetime_column].dt.quarter
                engineerDF["is_weekend"] = (engineerDF["day_of_week"] >= 5).astype(int)
                engineerDF["is_month_start"] = engineerDF[self.datetime_column].dt.is_month_start.astype(int)
                engineerDF["is_month_end"] = engineerDF[self.datetime_column].dt.is_month_end.astype(int)
                engineerDF["is_year_start"] = engineerDF[self.datetime_column].dt.is_year_start.astype(int)
                engineerDF["is_year_end"] = engineerDF[self.datetime_column].dt.is_year_end.astype(int)
                engineerDF["is_quarter_start"] = engineerDF[self.datetime_column].dt.is_quarter_start.astype(int)
                engineerDF["is_quarter_end"] = engineerDF[self.datetime_column].dt.is_quarter_end.astype(int)
                # Cyclic Features
                engineerDF["sin_day_of_week"] = np.sin(2 * np.pi * engineerDF["day_of_week"] / 7)
                engineerDF["cos_day_of_week"] = np.cos(2 * np.pi * engineerDF["day_of_week"] / 7)
                engineerDF["sin_month"] = np.sin(2 * np.pi * engineerDF["month_of_year"] / 12)
                engineerDF["cos_month"] = np.cos(2 * np.pi * engineerDF["month_of_year"] / 12)
                # Fourier Series
                engineerDF["fourier_sin_365"] = np.sin(2 * np.pi * engineerDF["day_of_year"] / 365)
                engineerDF["fourier_cos_365"] = np.cos(2 * np.pi * engineerDF["day_of_year"] / 365)
                engineerDF["fourier_sin_30"] = np.sin(2 * np.pi * engineerDF["day_of_month"] / 30)
                engineerDF["fourier_cos_30"] = np.cos(2 * np.pi * engineerDF["day_of_month"] / 30)
                if not scaler:
                    scaler = MinMaxScaler()
                    scaler.fit(engineerDF[engineeredFeatures])
                engineerDF[engineeredFeatures] = scaler.transform(engineerDF[engineeredFeatures])
                return engineerDF, scaler
            
            xgbFullData = pd.concat([self.trainData[[self.datetime_column, self.label]], self.testData[[self.datetime_column, self.label]]])
            xgbFullData, scaler = engineer_features(xgbFullData.copy())

            forecastRange = pd.date_range(start=self.forecastStartDatetime, end=self.forecastEndDatetime, freq=self.timeintervalFreq)
            forecastDF = pd.DataFrame()
            forecastDF[self.datetime_column] = forecastRange
            forecastDF, scaler = engineer_features(forecastDF.copy(), scaler)

            trainData = xgbFullData[(xgbFullData[self.datetime_column] >= self.trainData[self.datetime_column].min()) & (xgbFullData[self.datetime_column] <= self.trainData[self.datetime_column].max())]
            trainData = trainData.sample(frac=1).reset_index(drop=True)
            testData = xgbFullData[(xgbFullData[self.datetime_column] >= self.testData[self.datetime_column].min()) & (xgbFullData[self.datetime_column] <= self.testData[self.datetime_column].max())]
            xgbFullData = pd.concat([trainData, testData])
            xgbFullData.reset_index(drop=True, inplace=True)
            X_train = trainData[engineeredFeatures]
            y_train = trainData[self.label]
            X_test = testData[engineeredFeatures]
            y_test = testData[self.label]
            X_forecast = forecastDF[engineeredFeatures]
            d_train = xgb.DMatrix(X_train, label=y_train)
            d_test = xgb.DMatrix(X_test, label=y_test)
            d_full = xgb.DMatrix(xgbFullData[engineeredFeatures], label=xgbFullData[self.label])

            bestParameters = {}
            score = {}
            if self.tuneType == "Manual":
                bestParameters = self.parameters
            elif self.tuneType == "Auto":
                try:
                    objective = partial(
                        hypertune_xgbforecaster_model,
                        self = self,
                        y_test = y_test,
                        d_train = d_train,
                        d_test = d_test
                    )
                    early_stopper = EarlyStopper(threshold=0.01)
                    analysis = gp_minimize(
                        func=objective,
                        dimensions=self.parameters,
                        n_calls=config_instance.hypertune_sample_size,
                        callback=early_stopper
                    )
                    bestParameters["n_estimators"], bestParameters["max_depth"], bestParameters["learning_rate"], bestParameters["subsample"], bestParameters["colsample_bytree"], bestParameters["gamma"] = analysis.x
                    if analysis.fun != 9999:
                        score = {"Score %": round(-analysis.fun * 100, 2)}
                except:
                    pass
            if "n_estimators" not in bestParameters:    bestParameters["n_estimators"] = 1000
            
            model = xgb.train(bestParameters, d_full, num_boost_round=bestParameters["n_estimators"])
            forecastResult = model.predict(xgb.DMatrix(X_forecast, label=pd.Series([1 for _ in range(0, len(X_forecast))])))
            forecastDF[f"{self.label}_forecasted"] = forecastResult
            forecastDF.set_index(self.datetime_column, inplace=True)

            fullData = pd.concat([self.trainData, self.testData])
            fullData.set_index(self.datetime_column, inplace=True)
            labelIdx = fullData.columns.get_loc(self.label)
            resultDF = pd.concat([fullData.iloc[:, :labelIdx+1], forecastDF[f"{self.label}_forecasted"]], axis=0)
            resultDF = pd.concat([resultDF, fullData.iloc[:, labelIdx+1:]], axis=1)
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    resultDF[partitionColumn] = resultDF[partitionColumn].fillna(resultDF[partitionColumn].mode()[0])
            
            return resultDF, score
        except Exception as e:
            print(f"Error occurred while calling XGB Forecaster model: {e}")
            return pd.DataFrame(), {}

    def run_algorithm(self, rayDF):
        try:
            if len(rayDF) == 0:
                raise Exception("Empty DataFrame received for processing.")
            
            rayDF.sort_values(by=self.datetime_column, inplace=True)
            rayDF.drop_duplicates(subset=self.datetime_column, keep='first', inplace=True)
            rayDF.reset_index(drop=True, inplace=True)

            firstDate = rayDF[self.datetime_column].min()
            lastDate = rayDF[self.datetime_column].max()
            self.timeintervalFreq = pd.DatetimeIndex(rayDF[self.datetime_column]).inferred_freq
            if not self.timeintervalFreq:
                self.timeintervalFreq = rayDF[self.datetime_column].diff().mode()[0]
            
            rangeDS = pd.date_range(start=firstDate, end=lastDate, freq=self.timeintervalFreq)
            if len(rangeDS) == 0:
                raise Exception("No valid date range found in the DataFrame.")
            
            rayDF = rayDF.set_index(self.datetime_column)
            rayDF = rayDF.reindex(rangeDS, method='ffill')
            
            if self.groupbyFlag:
                for partitionColumn in self.config["partition_columns"]:
                    rayDF[partitionColumn] = rayDF[partitionColumn].fillna(rayDF[partitionColumn].mode()[0])
            
            stlInstance = STL(rayDF[self.label], seasonal=7, trend=None, robust=True)
            stlModel = stlInstance.fit()
            reconstructed_series = stlModel.trend + stlModel.seasonal + stlModel.resid.fillna(0)
            rayDF[self.label] = rayDF[self.label].combine_first(reconstructed_series)
            rayDF[self.label] = rayDF[self.label].interpolate(method='time').fillna(rayDF[self.label].bfill().ffill())
            rayDF.reset_index(inplace=True, drop=False, names=self.datetime_column)
            
            if self.config["split_method"] == "Percentage Split":
                trainTestSplit = self.config["train_test_split"].split(":")
                trainSplit = float(trainTestSplit[0]) / 100
                testSplit = float(trainTestSplit[1]) / 100
                sampleSize = len(rayDF)
                trainSize = int(trainSplit * sampleSize)
                testSize = sampleSize - trainSize
                trainStartDatetime = rayDF.at[0, self.datetime_column]
                trainEndDatetime = rayDF.at[trainSize - 1, self.datetime_column]
                testStartDatetime = rayDF.at[trainSize, self.datetime_column]
                testEndDatetime = rayDF.at[sampleSize - 1, self.datetime_column]
            elif self.config["split_method"] == "Time-based Split":
                trainStartDatetime = datetime.datetime.strptime(self.config["train_start_date"], self.datetimeFormat)
                trainEndDatetime = datetime.datetime.strptime(self.config["train_end_date"], self.datetimeFormat)
                testStartDatetime = datetime.datetime.strptime(self.config["test_start_date"], self.datetimeFormat)
                testEndDatetime = datetime.datetime.strptime(self.config["test_end_date"], self.datetimeFormat)
            else:
                raise ValueError("Invalid split method specified in configuration.")
            
            self.trainData = rayDF[(rayDF[self.datetime_column] >= trainStartDatetime) & (rayDF[self.datetime_column] <= trainEndDatetime)]
            self.testData = rayDF[(rayDF[self.datetime_column] >= testStartDatetime) & (rayDF[self.datetime_column] <= testEndDatetime)]
            if len(self.trainData)<4 or len(self.testData)<2:
                raise ValueError("Not enough data points for training-evaluation.")
            
            if self.config["forecast_method"] == 'Date Range':
                self.forecastStartDatetime = datetime.datetime.strptime(self.config["forecast_start_date"], self.datetimeFormat)
                self.forecastEndDatetime = datetime.datetime.strptime(self.config["forecast_end_date"], self.datetimeFormat)
            elif self.config["forecast_method"] == 'Next N Periods':
                self.forecastStartDatetime = testEndDatetime + datetime.timedelta(days=1)
                if self.config["forecast_period_unit"] == "Weeks":
                    self.config["forecast_periods"] = self.config["forecast_periods"] * 7
                elif self.config["forecast_period_unit"] == "Months":
                    self.config["forecast_periods"] = self.config["forecast_periods"] * 30
                self.forecastEndDatetime = self.forecastStartDatetime + datetime.timedelta(days=self.config["forecast_periods"])
            else:
                raise ValueError("Invalid forecast method specified in configuration.")
            
            if self.trainData[self.label].nunique()==1:
                result, score = self.call_single_value_predictor()
            elif self.algorithm_name == "ARIMA":
                result, score = self.call_arima_model()
            elif self.algorithm_name == "SARIMA":
                result, score = self.call_sarima_model()
            elif self.algorithm_name == "Prophet":
                result, score = self.call_prophet_model()
            elif self.algorithm_name == "Neural Prophet":
                result, score = self.call_neuralprophet_model()
            elif self.algorithm_name == "Attention-Based LSTM":
                result, score = self.call_ablstm_model() 
            elif self.algorithm_name == "XGBoost Forecaster":
                result, score = self.call_xgbforecaster_model()
            else:
                raise Exception("Unsupported Time Series Forecasting algorithm specified in configuration.")   
            if result.empty:
                raise Exception("Failed to run algorithm.")
            
            return result, score
        except Exception as e:
            print(f"Error occurred while running algorithm: {e}")
            return pd.DataFrame(), {}


