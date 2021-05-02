# %%

import os
import pandas as pd
import numpy as np
import random

# config parameters
CSV_PATH = "./oidv6-train-annotations-bbox.csv"
COLLECT_DATA_WRITE_FILE = "./data.txt"
NEW_CSV_PATH = "./train-annotations-bbox.csv"
IMAGE_DATA_FOLDER = "./open_image/"
CLASS_DESC_PATH = "./class-descriptions-boxable.csv"
CLASS_CODE = ['/m/01bfm9', '/m/01c648', '/m/01dwwc']

n_instance = 2000
train_test_split = 0.7

if not os.path.exists(IMAGE_DATA_FOLDER):
    os.mkdir(IMAGE_DATA_FOLDER)

# %%

n_train = int(n_instance * train_test_split)
df = pd.read_csv(CSV_PATH)

# %%

class_df = pd.read_csv(CLASS_DESC_PATH, names=['LabelName', 'name'])

CLASS_DICT = {}

for c in CLASS_CODE:
    CLASS_DICT[c] = class_df.loc[class_df.loc[:, 'LabelName'].values == c, 'name'].to_numpy()[0]

cond = False
for code in CLASS_CODE:
    cond |= (df.loc[:, 'LabelName'].values == code)

class_filter_df = df.loc[cond]
class_filter_df = class_filter_df.reset_index(drop=True)

img_id_unique_list = class_filter_df.loc[:, 'ImageID'].unique()

# %%

img_id_sample = np.random.choice(img_id_unique_list, size=n_instance, replace=False)
img_id_sample_train = img_id_sample[:n_train]
img_id_sample_test = img_id_sample[n_train:]


# %%

cond = False
for img_id in img_id_sample:
    cond |= (class_filter_df.loc[:, 'ImageID'] == img_id)

img_cls_filter_df = class_filter_df.loc[cond]


# %%

name = []
for i, row in img_cls_filter_df.iterrows():
    name.append(CLASS_DICT[row.loc["LabelName"]])

img_cls_filter_df.insert(3, "ClassName", name, True)
img_cls_filter_df.to_csv(NEW_CSV_PATH, index=False)

# %%

# Write files
with open(COLLECT_DATA_WRITE_FILE, "w") as f:
    f.writelines(['train/' + id + '\n' for id in img_id_sample_train])
    f.writelines(['train/' + id + '\n' for id in img_id_sample_test])

