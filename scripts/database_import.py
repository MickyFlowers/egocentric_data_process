import argparse

import os
import json
from supabase import create_client, Client
from tqdm import tqdm
# Supabase 配置
url = "http://192.168.3.208:54321"
key = "sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH"
supabase: Client = create_client(url, key)


def main(args):
    local_processd_dir = os.path.join(args.default_dataset_path, args.local_dataset_name, "processed")
    all_parts = []
    if os.path.exists(os.path.join(local_processd_dir, "render_meta.json")):
        all_parts.append(local_processd_dir)
    else:
        parts = [os.path.join(local_processd_dir, part) for part in os.listdir(local_processd_dir)]
        all_parts.extend(parts)
    # print(all_parts)
    for part in all_parts:
        render_meta_path = os.path.join(part, "render_meta.json")
        meta_path = os.path.join(part, "meta.json")
        with open(meta_path, "r") as f:
            ik_sample_ids = json.load(f)[0]['sample_ids']
        with open(render_meta_path, "r") as f:
            render_sample_ids = json.load(f)
        
        for ik_sample_id in tqdm(ik_sample_ids):
            data_path = os.path.join(part, "data", ik_sample_id + ".parquet").replace("/home/", "oss://")
            meta_data_path = os.path.join(part, "meta_data", ik_sample_id + ".json")
            video_path = os.path.join(part, "render", ik_sample_id + ".mp4").replace("/home/", "oss://")
            sample_id_meta = json.load(open(meta_data_path, "r"))
            
            path = sample_id_meta['video_path']
            if ik_sample_id not in render_sample_ids:
                result = {"agilex_ik_result_data_path": data_path, "agilex_ik_result_meta_data_path": meta_data_path.replace("/home/", "oss://")}
            else:
                result = {"agilex_ik_result_data_path": data_path, "agilex_ik_result_meta_data_path": meta_data_path.replace("/home/", "oss://"), "agilex_rendered_video_path": video_path.replace("/home/", "oss://")}
            # print(result)
            try:
                # pass
                supabase.table(args.table_name).update(result).eq("path", path).execute()
            except Exception as e:
                print(f"update {path} failed: {e}")
        
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_name", type=str, required=True)
    parser.add_argument("--remote_dataset_name", type=str, required=True)
    parser.add_argument("--table_name", type=str, default="egocentric_dataset_clips")
    parser.add_argument("--default_dataset_path", type=str, default="/home/ss-oss1/data/dataset/egocentric/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
