import os
import json


class Dataset:
    
    def __init__(self, info, data_name):
        
        self.data_name = data_name
        self.task_type = info.dataset2task[data_name]
        self.splits = json.load(open(os.path.join(info.DIR_INPUT, f'{data_name}.json')))