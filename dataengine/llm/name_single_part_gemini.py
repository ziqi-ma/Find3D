import vertexai
from vertexai import generative_models
from vertexai.generative_models import GenerativeModel, Part, Image
import os
import json
import time
from tqdm import tqdm
import threading
import traceback
import argparse
import pandas as pd
from dataengine.configs import DATA_ROOT

def query_gemini(prompt, image_paths):
    prompt_multimodal = [Part.from_image(Image.load_from_file(image_path)) for image_path in image_paths]
    prompt_multimodal.append(prompt)
    model = GenerativeModel(model_name="gemini-1.5-flash-001")
    response = model.generate_content(prompt_multimodal, safety_settings={
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH
    })
    ret_text = response.text
    return ret_text

def part_name_prompt():
    prompt = f"""
    What is the name of the part of the object that is masked out as purple? If you cannot find the part or are unsure, say unknown.
    Please only output the part name as one word or phrase.
    """
    return prompt

def parse_response(response):
    return response.lower().strip().replace("\n", "").replace("`", "").replace(":", "").replace("the answer is", "").replace("the purple part is", "").replace("the part marked out in purple is", "").replace("purple", "")

# batching results in low data quality, so will not batch
def query_part_dir(prompt, root_dir, exception_file):
    overall_dict_savepath = root_dir+"/masks/partnames.json"
    # if already queried, skip
    if not os.path.exists(root_dir): # rendering prob had issues
        return
    if os.path.exists(overall_dict_savepath):
        return
    all_image_paths = []
    if not os.path.exists(f"{root_dir}/masks"):
        exception_file.write(f"{root_dir}: no masks")
        return
    viewfolders = [f for f in os.listdir(f"{root_dir}/masks") if "." not in f]

    for viewfolder in viewfolders:
        view_dir = f"{root_dir}/masks/{viewfolder}" #... masks/view00
        image_paths = [f for f in os.listdir(view_dir) if f[-4:]==".png"]
        image_full_paths = [f"{viewfolder}/{image_path}" for image_path in image_paths]
        all_image_paths += image_full_paths
    
    part_dict = {}
    times = time.time()
    for image_path in all_image_paths:
        try:
            res = query_gemini(prompt, [f"{root_dir}/masks/{image_path}"])
            res_name = parse_response(res)
            part_dict[image_path] = res_name
        except Exception as e:
            if "Internal error" in str(e):
                print(traceback.format_exc())
                print("retry")
                try:
                    res = query_gemini(prompt, [f"{root_dir}/masks/{image_path}"])
                    res_name = parse_response(res)
                    part_dict[image_path] = res_name
                except Exception as e:
                    print(e)
                    exception_file.write(root_dir+f": {image_path} - query failed - {e}\n")
            elif "Quota exceeded" in str(e):
                # retry after 30s
                print("dial back")
                time.sleep(30)
                try:
                    res = query_gemini(prompt, [f"{root_dir}/masks/{image_path}"])
                    res_name = parse_response(res)
                    part_dict[image_path] = res_name
                except Exception as e:
                    print(e)
                    exception_file.write(root_dir+f": {image_path} - query failed - {e}\n")
            else:
                print(e)
                exception_file.write(root_dir+f": {image_path} - query failed - {e}\n")
    timee = time.time()
    print(timee-times)
    print((timee-times)/len(all_image_paths))
    print("-----------------------")
    with open(overall_dict_savepath, "w") as outfile:
        json.dump(part_dict, outfile)
    
    if len(all_image_paths) == 0:
        exception_file.write(f"{root_dir}: no image paths")
        return
    return


def query_object(obj_dir, exception_file):
    # first find out correct orientation
    prompt = part_name_prompt()
    query_part_dir(prompt, f"{obj_dir}/oriented", exception_file)
    return


def process_dirs(dirs, file_e):
    for obj_dir in tqdm(dirs):
        query_object(obj_dir, file_e)
        print(f"{obj_dir} done")


def run_endpoint_load(endpoint_idx):
    # further parallelize in 3 threads
    N_THREADS = 3
    parent_folder = f"{DATA_ROOT}/labeled/rendered"
    cur_df = pd.read_csv(f"{DATA_ROOT}/labeled/chunk_ids/chunk{chunk_idx}.csv")
    cur_df["path"] = cur_df["class"] + "_" + cur_df["uid"]
    child_dirs = cur_df["path"].tolist()
    full_dirs = [parent_folder+"/"+child_dir for child_dir in child_dirs]
    subchunk_size = len(full_dirs)//n_total_endpoints
    if len(full_dirs) != subchunk_size*n_total_endpoints:
        subchunk_size += 1

    file_e = open(f"name_part_exceptions_chunk{endpoint_idx}.txt", "a")
    times = time.time()
    project_id = proj_ids_map[endpoint_idx]
    vertexai.init(project=project_id, location=regions_map[endpoint_idx])
    cur_endpoint_chunk = full_dirs[endpoint_idx*subchunk_size:(endpoint_idx+1)*subchunk_size]
    
    n_rep = len(cur_endpoint_chunk) // N_THREADS
    threads = []
    for i in range(N_THREADS-1):
        threads.append(threading.Thread(target=process_dirs, args=(cur_endpoint_chunk[n_rep*i:n_rep*(i+1)],file_e,)))
    threads.append(threading.Thread(target=process_dirs, args=(cur_endpoint_chunk[n_rep*(N_THREADS-1):],file_e,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    timee = time.time()
    print(f"----------total time for {len(cur_endpoint_chunk)} in chunk {endpoint_idx}---------------------")
    print(timee-times)

if __name__ == "__main__":
    chunk_idx = 0 # change this to process all chunks
    n_total_endpoints = 10
    proj_ids_map = {
        0:"proj-name-0",
        1:"proj-name-1",
        2:"proj-name-2",
        3:"proj-name-3",
        4:"proj-name-4",
        5:"proj-name-5",
        6:"proj-name-6",
        7:"proj-name-7",
        8:"proj-name-8",
        9:"proj-name-9"
    }
    regions_map = {
        0:"us-central1",
        1:"us-west1",
        2:"us-west4",
        3:"us-south1",
        4:"us-east1",
        5:"us-east4",
        6:"us-east5",
        7:"europe-west4",
        8:"southamerica-east1",
        9:"northamerica-northeast1"
    }

    parser = argparse.ArgumentParser(description="Process an integer endpoint.")
    parser.add_argument('endpoint', type=int, help='An integer endpoint value')
    args = parser.parse_args()
    endpoint = args.endpoint
    
    run_endpoint_load(endpoint)

    


    
    

    
    
    