import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
import os
from tqdm import tqdm
import shutil
import pandas as pd
import argparse
from dataengine.configs import DATA_ROOT

def query_gemini(prompt, image_paths):
    prompt_multimodal = [Part.from_image(Image.load_from_file(image_path)) for image_path in image_paths]
    prompt_multimodal.append(prompt)
    model = GenerativeModel(model_name="gemini-1.5-flash-001")
    response = model.generate_content(prompt_multimodal)
    ret_text = response.text
    return ret_text

def construct_prompt():
    prompt = f"""
    For each image, is the object in an orientation that is usually seen? Please answer yes or no for each image.  
    """
    return prompt

def parse_response(response):
    n_yes = response.lower().count("yes")
    n_no = response.lower().count("no")
    if n_yes + n_no == 0:
        return 0
    return n_yes/(n_yes+n_no)


def query_uid(root_dir):
    # skip if already annotated
    if not os.path.exists(root_dir): # rendering prob had issues
        return
    if "orientation.txt" in os.listdir(root_dir):
        return
    prompt = construct_prompt()
    orientations = ["norotate", "front2top", "flip"]
    correct_orientation = "none"
    max_yes_ratio = -1
    for orientation in orientations:
        img_paths = [f"{root_dir}/{orientation}/imgs/{i:02d}.jpeg" for i in range(10)]
        try:
            gemini_response = query_gemini(prompt,img_paths)
            yes_ratio = parse_response(gemini_response)
            if yes_ratio > max_yes_ratio:
                max_yes_ratio = yes_ratio
                correct_orientation = orientation
        except Exception:
            # if fails, set norotate as default
            correct_orientation = "norotate"
    with open(f'{root_dir}/orientation.txt', 'w') as f:
        f.write(correct_orientation)
    # delete the other orientations
    for orientation in orientations:
        if orientation != correct_orientation:
            shutil.rmtree(f"{root_dir}/{orientation}")
    os.rename(f"{root_dir}/{correct_orientation}", f"{root_dir}/oriented")
    return

def process_endpoint(endpoint_idx):
    project_id = proj_ids_map[endpoint_idx]
    vertexai.init(project=project_id, location=regions_map[endpoint])
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    
    full_dirs = [parent_folder+"/"+child_dir for child_dir in child_dirs]
    subchunk_size = len(full_dirs)//n_total_endpoints
    if len(full_dirs) != subchunk_size*n_total_endpoints:
        subchunk_size += 1
    cur_dirs = full_dirs[endpoint*subchunk_size:(endpoint+1)*subchunk_size]

    file_e = open(f"orientation_exceptions-{endpoint}.txt", "a")
    for dir in tqdm(cur_dirs):
        try:
            query_uid(dir)
        except Exception:
            file_e.write(f"{dir}\n")
    file_e.close()


if __name__ == "__main__":
    chunk_idx = 0

    n_total_endpoints = 3
    proj_ids_map = {
        0:"proj-name-0",
        1:"proj-name-1",
        2:"proj-name-2"
    }
    regions_map = {
        0:"us-central1",
        1:"us-west1",
        2:"us-west4"
    }
    parser = argparse.ArgumentParser(description="Process an integer endpoint.")
    parser.add_argument('endpoint', type=int, help='An integer endpoint value')
    args = parser.parse_args()
    endpoint = args.endpoint
    process_endpoint(endpoint)

    