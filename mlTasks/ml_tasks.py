import os
import ray
import math
import psutil
import datetime
import numpy as np
import pandas as pd
from configurations.config import config_instance

@ray.remote
def run_algorithm_in_ray(config, rayDFRef):
    if config["ml_task"] == "Regression":
        from mlTasks.regressionModels.regression_model import RegressionInRay
        runAlgoInstance = RegressionInRay(config)
    elif config["ml_task"] == "Outlier Detection":
        from mlTasks.anomalyModels.anomaly_model import AnomalyInRay
        runAlgoInstance = AnomalyInRay(config)
    elif config["ml_task"] == "Time Series Forecasting":
        from mlTasks.forecastingModels.forecasting_model import ForecastingInRay
        runAlgoInstance = ForecastingInRay(config)
    elif config["ml_task"] == "Classification":
        from mlTasks.classificationModels.classification_model import ClassificationInRay
        runAlgoInstance = ClassificationInRay(config)
    else:
        raise ValueError("Invalid machine learning task specified in configurationJSON.")
    
    result, score = runAlgoInstance.run_algorithm(rayDFRef)
    del runAlgoInstance
    return result, score

@ray.remote
class GetResource:
    def getResFunc(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        return (cpu_percent + memory_percent)

class MLTasks:
    def __init__(self, configurationJSON, rayDFList):
        self.configurationJSON = configurationJSON
        self.rayDFList = rayDFList
        self.partitionCount = len(self.rayDFList)
        self.totalRayNodesAvailable = 1
        self.finalResult = []
        self.successCount = 0
        self.failedCount = 0

    def update_progress(self, score, force_complete=False):
        progress_file = os.path.join(os.getcwd(), "Output", "progress.csv")
        if os.path.exists(progress_file):
            read_progress_df = pd.read_csv(progress_file)
            read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Progress'] = f"{(self.successCount + self.failedCount)/self.partitionCount*100:.2f}%"
            if force_complete:
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Progress'] = "100.00%"
            read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'End_Time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Execution_Time'] = str(datetime.datetime.now() - self.configurationJSON["timestamp"])[:-3]
            if self.successCount == self.partitionCount:
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = "Successfully Completed"
            elif self.failedCount == self.partitionCount:
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = "Failed"
            elif self.successCount + self.failedCount == self.partitionCount:
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Status'] = f"Partially Completed"
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Remarks'] = f"{self.successCount} Success, {self.failedCount} failed"
            if score:
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Metric_Name'] = list(score.keys())[0]
                read_progress_df.loc[read_progress_df['Model_ID'] == self.configurationJSON["model_id"], 'Metric_Value'] = list(score.values())[0]
            read_progress_df.to_csv(progress_file, index=False)

    def sort_by_best_ray_nodes(self):
        try:
            rayNodeList = ray.nodes()
            nodeDict = {}
            for node in rayNodeList:
                if node["Alive"] and node["Resources"]["CPU"] > 0:
                    nodeIP = node["NodeManagerAddress"].split(":")[0]
                    nodeActor = GetResource.options(resources={f"node:{nodeIP}":0.001}).remote()
                    resFunc_future = nodeActor.getResFunc.remote()
                    ready, _ = ray.wait([resFunc_future], timeout=15)
                    if ready:
                        nodeResourceUsage = ray.get(resFunc_future)
                    else:
                        nodeResourceUsage = 100 # 100% usage. No resource free in the worker node
                    ray.kill(nodeActor)
                    nodeDict[nodeIP] = nodeResourceUsage
            sortedNodeList = dict(sorted(nodeDict.items(), key=lambda item: item[1]))
            bestNodeList = [nodeKey for nodeKey in sortedNodeList.keys() if sortedNodeList[nodeKey] < 100]
            return bestNodeList
        except Exception as e:
            print(f"Exception Occurred while sorting ray nodes. \nError Reads: {e}")
            return []

    def execute(self):
        print(f"\n\n\n\n\n\n\n---x---\n\n\n\n\n\n\n::: NOW EXECUTING :::")
        rayNodeList = self.sort_by_best_ray_nodes()
        rayNodeList = [False] if not rayNodeList else rayNodeList

        resources = {}
        resourcesList = []
        if (rayNodeList) and (rayNodeList[0] != False):
            for node in rayNodeList:
                for nodes in ray.nodes():
                    if node in nodes["NodeManagerAddress"] and nodes["Alive"] is True:
                        use_resource = (nodes["Resources"]["CPU"])
                        use_resource = round(config_instance.rayResourcePerAlgo/use_resource, 3)
                        resources[f"node:{node}"] = use_resource
                        resourcesList.append(resources)
        if not resourcesList:
            resourcesList = [{}]
        self.totalRayNodesAvailable = len(resourcesList)

        batchSize = config_instance.batchSize
        if batchSize > self.partitionCount:
            batchSize = self.partitionCount
        
        batchIDs = []
        batchIndex = 0
        for rayDFRef in self.rayDFList[:batchSize]:
            batchIDs.append(run_algorithm_in_ray.options(resources=resourcesList[batchIndex%self.totalRayNodesAvailable]).remote(self.configurationJSON, rayDFRef))
            batchIndex += 1
        
        while len(batchIDs):
            doneID, batchIDs = ray.wait(batchIDs)
            completedTaskID = doneID[0]
            try:
                if self.partitionCount > batchIndex:
                    batchIDs.append(run_algorithm_in_ray.options(resources=resourcesList[batchIndex%self.totalRayNodesAvailable]).remote(self.configurationJSON, self.rayDFList[batchIndex]))
                    batchIndex += 1
            except Exception as e:
                print(f"Exception Occurred while executing forecasting model. \nError Reads: {e}")
                continue

            result, score = ray.get(completedTaskID)
            if not result.empty:
                self.finalResult.append(result)
                self.successCount += 1
            else:
                self.failedCount += 1
            self.update_progress(score=score)

        self.finalResult = pd.concat(self.finalResult)
        if not self.finalResult.empty:
            self.finalResult.reset_index(drop=True, inplace=True)
            self.finalResult.to_csv(os.path.join(os.getcwd(), "Output", f"{self.configurationJSON['model_id']}.csv"), index=False)
        self.update_progress(score=score, force_complete=True)
        
        print(f"---X--- {self.configurationJSON['model_id']} COMPLETED ---X---")
            