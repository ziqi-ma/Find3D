import os
import pandas as pd
from dataengine.configs import DATA_ROOT

def train_test_split(train_ratio, data_root):
    legal_ids = pd.read_csv(f"{data_root}/legal_uids.csv").sample(frac=1)
    legal_ids["path"] = legal_ids["class"] + "_" + legal_ids["uid"]

    n_train = int(len(legal_ids)*train_ratio) # 90 10 split
    train_paths = legal_ids["path"].tolist()[:n_train] # 27204
    val_paths = legal_ids["path"].tolist()[n_train:] # 3023

    # save
    with open(f"{data_root}/labeled/split/train.txt", 'w') as f:
        for line in train_paths:
            f.write(f"{line}\n")

    with open(f"{data_root}/labeled/split/val.txt", 'w') as f:
        for line in val_paths:
            f.write(f"{line}\n")


def keep_valid_objects(data_root):
    # filter out and keep only valid (has mask) ids
    with open(f"{data_root}/labeled/split/train.txt", 'r') as f:
        legal_train_ids = f.read().splitlines()
    new_train_ids = []
    for nameid in legal_train_ids:
        id = nameid.split("_")[-1]
        if not os.path.exists(f"{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/allmasks.pt"):
            continue
        if not os.path.exists(f"{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask_labels.txt"):
            continue
        if not os.path.exists(f"{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask2points.pt"):
            continue
        if not os.path.exists(f"{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask2view.pt"):
            continue
        if not os.path.exists(f"{data_root}/labeled/points/{id}"):
            continue
        new_train_ids.append(nameid)
    
    os.remove(f"{data_root}/labeled/split/train.txt")
    with open(f"{data_root}/labeled/split/train.txt", 'w') as f:
        for line in new_train_ids:
            f.write(f"{line}\n")
    
    with open(f"{data_root}/labeled/split/val.txt", 'r') as f:
        legal_val_ids = f.read().splitlines()
    new_val_ids = []

    for nameid in legal_val_ids:
        id = nameid.split("_")[-1]
        if not os.path.exists(f'{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/allmasks.pt'):
            continue
        if not os.path.exists(f'{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask_labels.txt'):
            continue
        if not os.path.exists(f'{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask2points.pt'):
            continue
        if not os.path.exists(f'{data_root}/labeled/rendered/{nameid}/oriented/masks/merged/mask2view.pt'):
            continue
        if not os.path.exists(f"{data_root}/labeled/points/{id}"):
            continue
        new_val_ids.append(nameid)
    
    os.remove(f'{data_root}/labeled/split/val.txt')
    with open(f'{data_root}/labeled/split/val.txt', 'w') as f:
        for line in new_val_ids:
            f.write(f"{line}\n")


if __name__ == "__main__":
    train_test_split(0.9, DATA_ROOT)
    keep_valid_objects(DATA_ROOT)


    




    





     
    


