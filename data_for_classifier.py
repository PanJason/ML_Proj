# prepare data for the classifier

import numpy as np 
import json

dataset_path = ''
json_path = 'fracture/annotations/anno_train.json'

with open(json_path) as f:
    instances_train = json.load(f)

anno_train = instances_train['annotations']
print(anno_train[0])



