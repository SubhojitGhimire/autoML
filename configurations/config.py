class Config:
    def __init__(self):
        self.hostIP = "127.0.0.1"
        self.hostPort = 8080

        self.rayAddress = None # ("ray://<head_node_ip>:<gcs_port>"); "auto" for automatic detection; "localhost" for localhost
        self.rayResourcePerAlgo = 2
        
        self.batchSize = 3
        self.hypertune_sample_size = 30


config_instance = Config()