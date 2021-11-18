#%%
import os
import json

TRAIN_TEST_PATH = "./challenge-data/train_test.txt"
TRAIN_JSON = "./challenge-data/train_test.json"
#%%
data = []
with open(TRAIN_TEST_PATH) as train_data:
    for line in train_data:
        strings = line.split('\t')
        pair = {1: strings[0].split(), 0: strings[1].split()}
        data.append(pair)

#%%
with open(TRAIN_JSON, 'w') as train_json:
    json.dump(data, train_json)
        
# %%
with open(TRAIN_JSON) as train_json:
    json_read = json.load(train_json)
    print(json_read[-6])
# %%
