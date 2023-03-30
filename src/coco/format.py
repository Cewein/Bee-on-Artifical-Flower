# % a file the return a json object from a coco format file

import json
import numpy as np

def openCocoFile(path: str) -> json:
    data = None
    with open(path) as data_file:    
        data = json.load(data_file)

    return data

# %
def jsonToArray(jsonDict: dict) -> tuple((list, list, np.array)):
    categories = []

    for i in range(len(jsonDict['categories'])):
        categories.append(jsonDict['categories'][i]['name'])

    images = []

    for i in range(len(jsonDict['images'])):
        images.append(jsonDict['images'][i]['file_name'])

    annotations = []

    for i in range(len(jsonDict['annotations'])):
            annotations.append([
            jsonDict['annotations'][i]['image_id'] - 1,
            jsonDict['annotations'][i]['category_id'] - 1,
            jsonDict['annotations'][i]['bbox'][0],
            jsonDict['annotations'][i]['bbox'][1],
            jsonDict['annotations'][i]['bbox'][2],
            jsonDict['annotations'][i]['bbox'][3]
            ])
    
    annotations = sorted(annotations, key=lambda l:l[0])

    return categories,images,np.array(annotations, dtype=np.int32)