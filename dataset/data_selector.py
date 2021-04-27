# %%

import os
import pandas as pd
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

new_df = df.loc[cond]
new_df = new_df.reset_index(drop=True)


assert n_instance <= new_df.shape[0], "Random sampling need to have large enough size or small enough n_instance"

r_list = random.sample(range(0, new_df.shape[0] - 1), n_instance)
r_list_train = r_list[:n_train]
r_list_test = r_list[n_train:]



new_df = new_df.iloc[r_list]


name = []
for i, row in new_df.iterrows():
    name.append(CLASS_DICT[row.loc["LabelName"]])

new_df.insert(3, "ClassName", name, True)


new_df.to_csv(NEW_CSV_PATH, index=False)


# Write files
with open(COLLECT_DATA_WRITE_FILE, "w") as f:
    f.writelines(['train/' + new_df.loc[r, 'ImageID'] + '\n' for r in r_list_train])
    f.writelines(['train/' + new_df.loc[r, 'ImageID'] + '\n' for r in r_list_test])


