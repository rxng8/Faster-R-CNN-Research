# %%

import os
import pandas as pd
import random

# config parameters
CSV_PATH = "./oidv6-train-annotations-bbox.csv"
COLLECT_DATA_WRITE_FILE = "./data.txt"
NEW_CSV_PATH = "./train-annotations-bbox.csv"
IMAGE_DATA_FOLDER = "./open_image/"
n_instance = 2000
train_test_split = 0.7

if not os.path.exists(IMAGE_DATA_FOLDER):
    os.mkdir(IMAGE_DATA_FOLDER)

# %%

n_train = int(n_instance * train_test_split)
df = pd.read_csv(CSV_PATH)

r_list = random.sample(range(0, df.shape[0] - 1), n_instance)
r_list_train = r_list[:n_train]
r_list_test = r_list[n_train:]

with open(COLLECT_DATA_WRITE_FILE, "w") as f:
    f.writelines(['train/' + df.loc[r, 'ImageID'] + '\n' for r in r_list_train])
    f.writelines(['test/' + df.loc[r, 'ImageID'] + '\n' for r in r_list_test])

new_df = df.iloc[r_list]
new_df.to_csv(NEW_CSV_PATH, index=False)



