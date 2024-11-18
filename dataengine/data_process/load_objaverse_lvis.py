### load data from objaverse
import objaverse
import pandas as pd
import json
from dataengine.configs import WONDER3D_UIDS_PATH, DATA_ROOT, HOLDOUT_CLASSES_PATH, HOLDOUT_SEENCLASS_IDS_PATH, HOLDOUT_SHAPENETPART_IDS_PATH

# we save as chunks of chunk_size and can parallely process chunks
def get_label_ids(chunk_size, data_root): 
    # first get LVIS categories
    lvis_annotations = objaverse.load_lvis_annotations()
    lvis_uids = []
    lvis_uids_corr = []
    for l in lvis_annotations:
        lvis_uids += lvis_annotations[l]
        for uid in lvis_annotations[l]:
            lvis_uids_corr.append([uid, l])
    all_classes = pd.DataFrame(data=lvis_uids_corr, columns=["uid", "class"])
    all_classes.to_csv(data_root+"/metadata.csv")
    
    # check filtered by wonder3d
    with open(WONDER3D_UIDS_PATH) as f:
        filtered_uid_list = json.load(f)
    filtered = all_classes[all_classes['uid'].isin(filtered_uid_list)]

    val_counts = filtered["class"].value_counts()
    kept_categories = val_counts[val_counts["count"] > 12]["class"].tolist()

    # exclude holdout classes
    with open(HOLDOUT_CLASSES_PATH, 'r') as f:
        holdout_ids = f.read().splitlines()

    classes_to_exclude = all_classes[all_classes['uid'].isin(holdout_ids)]['class'].tolist()
    filtered = filtered[filtered["class"].isin(kept_categories)]
    filtered = filtered[~filtered["class"].isin(classes_to_exclude)]

    # exclude holdout instances
    ids_to_exclude = []
    with open(HOLDOUT_SEENCLASS_IDS_PATH, 'r') as f:
        holdout_trainclass_new = f.read().splitlines()
    ids_to_exclude += holdout_trainclass_new
    with open(HOLDOUT_SHAPENETPART_IDS_PATH, 'r') as f:
        holdout_sn_ids = f.read().splitlines()
    ids_to_exclude += holdout_sn_ids
    filtered = filtered[~filtered["class"].isin(classes_to_exclude)]
    filtered = filtered[~filtered["uid"].isin(ids_to_exclude)]
    filtered.to_csv(f"{data_root}/legal_uids.csv", index=False)

    # chunk these
    n_chunks = len(filtered)//chunk_size+1
    for i in range(n_chunks):
        curchunk = filtered.iloc[i*chunk_size:(i+1)*chunk_size]
        curchunk.to_csv(f"{data_root}/labeled/chunk_ids/chunk{i}.csv", index=False)
    return n_chunks


def download_obj_lvis(chunk_id, data_root):
    try_uids = pd.read_csv(f"{data_root}/labeled/chunk_ids/chunk{chunk_id}.csv")["uid"].tolist()
    objaverse._VERSIONED_PATH = f"{data_root}/labeled/glbs"
    objaverse.load_objects(
        uids=try_uids,
        download_processes=32
    )

if __name__ == "__main__":
    n_chunks = get_label_ids(7000, DATA_ROOT)
    for i in range(n_chunks):
        download_obj_lvis(i, DATA_ROOT)
