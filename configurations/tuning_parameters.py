arima_paramSpace = {
    "p": [0, 5, None, "int"],
    "d": [0, 2, None, "int"],
    "q": [0, 5, None, "int"],
}
sarima_paramSpace = {
    "p": [0, 5, None, "int"],
    "d": [0, 2, None, "int"],
    "q": [0, 5, None, "int"],
    "P": [0, 2, None, "int"],
    "D": [0, 1, None, "int"],
    "Q": [0, 2, None, "int"],
    "s": [2, 12, None, "int"]
}
prophet_paramSpace = {
    "seasonality_mode": [["additive", "multiplicative"], "category"],
    "changepoint_prior_scale": [0.001, 0.5, 0.001, "float"],
    "seasonality_prior_scale": [0.01, 10.0, 0.01, "float"],
    "holidays_prior_scale": [0.01, 10.0, 0.01, "float"],
    "weekly_seasonality": [[True, False], "category"],
    "daily_seasonality": [[True, False], "category"],
    "yearly_seasonality": [[True, False], "category"]
}
neuralProphet_paramSpace = {
    "learning_rate": [0.001, 0.1, None, "float"],
    "yearly_seasonality": [[True, False], "category"],
    "weekly_seasonality": [[True, False], "category"],
    "daily_seasonality": [[True, False], "category"],
    "seasonality_mode": [["additive", "multiplicative"], "category"],
    "changepoints_range": [0.8, 1, None, "float"]
}
attentionBasedLSTM_paramSpace = {
    "LSTM Units": [[2 ** i for i in range(8)], "category"],
    "Attention Units": [[2 ** i for i in range(8)], "category"]
}
xgboostRegression_paramSpace = {
    "n_estimators": [10, 1000, 100, "int"],
    "max_depth": [3, 10, None, "int"],
    "learning_rate": [0.01, 0.3, 0.01, "float"],
    "subsample": [0.5, 1.0, 0.05, "float"],
    "colsample_bytree": [0.5, 1.0, 0.05, "float"],
    "gamma": [0, 100, 5, "int"],
}
randomForest_paramSpace = {
    "n_estimators": [50, 500, 50, "int"],
    "criterion": [["squared_error", "absolute_error", "friedman_mse", "poisson"], "category"],
    "max_depth": [3, 50, None, "int"],
    "min_samples_split": [2, 20, 1, "int"],
    "min_samples_leaf": [1, 10, 1, "int"],
    "max_features": [["sqrt", "log2", None], "category"]
}
linearRegression_paramSpace = {
    "fit_intercept": [[True, False], "category"],
    "copy_X": [[True, False], "category"],
    "tol": [1e-8, 1e-3, None, "float"],
    "positive": [[True, False], "category"]
}
SVR_paramSpace = {
    "kernel": [["linear", "poly", "rbf", "sigmoid"], "category"],
    "degree": [2, 5, 1, "int"],  # Only used if kernel="poly"
    "coef0": [0.0, 1.0, 0.1, "float"],  # relevant for poly/sigmoid
    "tol": [1e-5, 1e-2, None, "float"],
    "C": [0.1, 10.0, 0.1, "float"],
    "epsilon": [0.01, 1.0, 0.01, "float"],
    "shrinking": [[True, False], "category"],
    "cache_size": [100, 1000, 100, "int"],
    "max_iter": [-1, 5000, 100, "int"]
}
linearSVC_paramSpace = {
    "penalty": [["l1", "l2"], "category"],
    "tol": [[1e-4, 1e-3, 1e-2], "category"],
    "C": [[0.01, 0.1, 1.0, 10.0, 100.0], "category"],
    "fit_intercept": [[True, False], "category"],
    "intercept_scaling": [[0.1, 1, 10], "category"],
    "class_weight": [[None, "balanced"], "category"],
    "max_iter": [500, 5000, 500, "int"]
}
logisticRegression_paramSpace = {
    "penalty": [["l1", "l2", "elasticnet", None], "category"],
    "solver": [["liblinear", "saga", "lbfgs", "newton-cg"], "category"],
    "C": [[0.01, 0.1, 1.0, 10.0, 100.0], "category"],
    "tol": [[1e-4, 1e-3, 1e-2, "category"]],
    "fit_intercept": [[True, False], "category"],
    "intercept_scaling": [[0.1, 1, 10], "category"],
    "class_weight": [[None, "balanced"], "category"],
    "max_iter": [500, 5000, 500, "int"],
    "multi_class": [["auto", "ovr", "multinomial"], "category"],
    "l1_ratio": [[None, 0.0, 0.5, 1.0], "category"]
}
lof_paramSpace = {
    'n_neighbors': [5, 50, 5, "int"],
    'algorithm': [['auto', 'ball_tree', 'kd_tree', 'brute'], "category"],
    'leaf_size': [10, 50, 10, "int"],
    'metric': [['minkowski', 'euclidean', 'manhattan', 'chebyshev'], "category"],
    'p': [1, 4, 1, "int"],
    'contamination': [['auto', 0.01, 0.05, 0.1, 0.2], "category"]
}
oneclassSVM_paramSpace = {
    'kernel': [['linear', 'poly', 'rbf', 'sigmoid'], "category"],
    'gamma': [['scale', 'auto', 0.01, 0.1, 1], "category"],
    'degree': [2, 6, 1, "int"],
    'coef0': [[0.0, 0.1, 0.5, 1.0], "category"],
    'tol': [[1e-4, 1e-3, 1e-2], "category"],
    'nu': [[0.01, 0.05, 0.1, 0.2, 0.5], "category"],
    'shrinking': [[True, False], "category"],
    'max_iter': [-1, 10000, 1000, "int"]
}
catBoost_paramSpace = {
    "iterations": [100, 1000, 200, "int"],
    "learning_rate": [0.01, 0.3, 0.01, "float"],
    "depth": [3, 10, None, "int"],
    "l2_leaf_reg": [1, 10, 2, "int"],
    "random_strength": [[0.0, 1.0, 2.0, 5.0], "category"],
    "bagging_temperature": [0.0, 2.0, 0.5, "float"],
    "border_count": [[32, 64, 128], "category"],
    "grow_policy": [["SymmetricTree", "Depthwise", "Lossguide"], "category"],
    "subsample": [[0.7, 0.8, 1.0], "category"],
    "colsample_bylevel": [[0.5, 0.8, 1.0], "category"]
}


def get_tune_parameters(algorithmName):
    if algorithmName == "ARIMA":
        return arima_paramSpace
    elif algorithmName == "SARIMA":
        return sarima_paramSpace
    elif algorithmName == "Prophet":
        return prophet_paramSpace
    elif algorithmName == "Neural Prophet":
        return neuralProphet_paramSpace
    elif algorithmName == "Attention-Based LSTM":
        return attentionBasedLSTM_paramSpace
    elif "XGBoost" in algorithmName:
        return xgboostRegression_paramSpace
    elif "Random Forest" in algorithmName:
        return randomForest_paramSpace
    elif algorithmName == "Linear Regression":
        return linearRegression_paramSpace
    elif algorithmName == "SVM Regressor":
        return SVR_paramSpace
    elif algorithmName == "Linear SVM Classifier":
        return linearSVC_paramSpace
    elif algorithmName == "Logistic Regression":
        return logisticRegression_paramSpace
    elif algorithmName == "Local Outlier Factor":
        return lof_paramSpace
    elif algorithmName == "One-Class SVM":
        return oneclassSVM_paramSpace
    elif algorithmName == "CatBoost Outlier Detector":
        return catBoost_paramSpace
    else:
        raise Exception("Unsupported Algorithm Name")

