import ray
import pandas as pd
from configurations.config import config_instance
from configurations.tuning_parameters import get_tune_parameters

import os
import json
import datetime
import requests
import threading
import numpy as np

from skopt.space import Integer, Real, Categorical

def start_model_training(configurationJSON, exogDF):
    print("Starting model training with configuration:", configurationJSON)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env={"working_dir": os.getcwd()}) if config_instance.rayAddress == "localhost" else ray.init(address=config_instance.rayAddress, ignore_reinit_error=True, runtime_env={"working_dir": os.getcwd()})
    
    rayDFList = []
    if configurationJSON["partition_columns"]:
        groupedExogDF = exogDF.groupby(configurationJSON["partition_columns"])
        # for groupName in groupedExogDF.groups:
        #     groupDF = groupedExogDF.get_group(groupName)
        for groupName, groupDF in groupedExogDF:
            rayDFList.append(ray.put(groupDF))
    else:
        rayDFList = [ray.put(exogDF)]

    if configurationJSON["hyperparameter_tuning"] == "Manual":
        parameters = {}
        manual_param = configurationJSON["manual_params"]
        for parameter in manual_param:
            if manual_param[parameter]["type"] == "int":
                parameters[parameter] = int(manual_param[parameter]["value"])
            elif manual_param[parameter]["type"] == "float":
                parameters[parameter] = float(manual_param[parameter]["value"])
            elif manual_param[parameter]["type"] == "bool":
                parameters[parameter] = bool(manual_param[parameter]["value"])
            elif manual_param[parameter]["type"] == "category":
                parameters[parameter] = str(manual_param[parameter]["value"])
                if parameters[parameter] == "True" | parameters[parameter] == "False":
                    parameters[parameter] = bool(parameters[parameter])
    elif configurationJSON["hyperparameter_tuning"] == "Auto":
        sample_space = []
        tune_params = get_tune_parameters(configurationJSON["algorithm"])
        for sample in tune_params:
            if tune_params[sample][-1] == "int":
                if not tune_params[sample][2]:
                    sample_space.append(Integer(tune_params[sample][0], tune_params[sample][1], name=sample))
                else:
                    sample_space.append(Categorical(np.arange(tune_params[sample][0], tune_params[sample][1], tune_params[sample][2], dtype=int), name=sample))
            elif tune_params[sample][-1] == "float":
                if not tune_params[sample][2]:
                    sample_space.append(Real(tune_params[sample][0], tune_params[sample][1], name=sample))
                else:
                    sample_space.append(Categorical(np.arange(tune_params[sample][0], tune_params[sample][1], tune_params[sample][2]), name=sample))
            elif tune_params[sample][-1] == "category":
                sample_space.append(Categorical(tune_params[sample][0], name=sample))
        parameters = sample_space
    else:
        raise ValueError("Invalid hyperparameter tuning method specified in configuration.")
    configurationJSON["parameters"] = parameters
    print("After processed configuration:", configurationJSON["parameters"])
    
    from mlTasks import ml_tasks
    mlTasksInstance = ml_tasks.MLTasks(configurationJSON, rayDFList)
    mlTasksInstance.execute()
    
