# Data Engine
The Data engine automatically annotates 3D assets from Objaverse, and creates training data for Find3D.

# Prerequisites:
The data engine relies on SAM and Gemini. Please first download necessary checkpoints/configs and fill in their paths in configs.py. Please also create a vertex AI account to query Gemini.

1. Download SAM checkpoint [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

2. Download a json containing high-quality Objaverse asset uid's selected by Wonder3D [here](https://github.com/xxlong0/Wonder3D/blob/main/data_lists/lvis_uids_filter_by_vertex.json).

3. Fill in the path variables in [`configs.py`](configs.py). This includes the two downloaded paths above, as well as a `DATA_ROOT` which is where all the generated data will be stored.

# Environment
```
cd dataengine
conda create --name <env> --file requirements.txt
export PYTHONPATH=[path to Find3D]
```
# The data pipeline:
1. [`dataengine/data_process/load_objavers_lvis.py`](data_process/load_objaverse_lvis.py) downloads Objaverse assets.

2. [`dataengine/rendering/render_2d.py`](rendering/render_2d.py) uses pytorch3d to render out k views per object for multiple orientations. The rendering produces point-face correspondence, which is saved at this step.

3. [`dataengine/llm/query_gemini_orientation.py`](llm/query_gemini_orientation.py) queries Gemini to obtain the most commonly-seen orientation of each object. This is helpful for obtaining higher-quality Gemini annotations. This scripts deletes the other orientations and puts all data for an object under a directory named "oriented". During training, we do not abide by this orientation. This script parallelizes across multiple Gemini endpoints.

4. Sample points from the Objaverse meshes using pytorch3d, and keep the point correspondence to faces. This requires a small customization of the sample_point_from_meshes function in `pytorch3d/ops/sample_points_from_meshes.py`. We provide this customized function in [`dataengine/py3d_customization/sample_points_from_meshes.py`](py3d_customization/sample_points_from_meshes.py).

5. [`dataengine/seg2d/get_sam_masks.py`](seg2d/get_sam_masks.py) calls SAM and gets (an undetermined number of) masks that satisfy criteria. This script also produces overlay of masks on the original image, with the mask marked as purple.

6. [`dataengine/llm/name_single_part_gemini.py`](llm/name_single_part_gemini.py) prompts Gemini with the overlay images to get each part's name by Gemini. This script parallelizes across multiple Gemini endpoints. 

7. [`dataengine/seg2d/merge_masks.py`](seg2d/merge_masks.py) creates merged mask (N_CUR_MASK,H,W) with view correspondence (see details in [`merge_masks.py`](seg2d/merge_masks.py)).

8. [`dataengine/label3d/label_mask2pt.py`](label3d/label_mask2pt.py) gets the mask-to-point correspondence (used in contrastive loss)

We note that we do not directly save the point features because they are too large (estimate >=50TB). With steps 6 and 7, we obtain all data needed to for the contrastive objective:
* `allmasks.pt` (this is n_masks*h*w binary)
* `mask2view.pt` (this is shape (n_masks,) each entry is a view id 0-9)
* `mask2points.pt` (this is n_masks*n_pts binary)
* `mask_labels.txt` for each object

9. [`dataengine/data_process/train_test_split.py`](data_process/train_test_split.py) performs train-test split and filters out insufficiently-labeled data.

We note that all scripts except `load_objaverse_lvis.py` and `train_test_split.py` contain a chunk_id variable to enable incremental processing or parallelization. As with most large-scale workloads, we recommend incremental processing per chunk and manual inspection to avoid propagating errors due to machine/memory failures.