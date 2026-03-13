import argparse
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _process_sample(input_dir: str, prefix: str, sample_id: str) -> str:
    meta_data_path = os.path.join(input_dir, "meta_data", sample_id + ".json")
    with open(meta_data_path, "r") as f:
        data = json.load(f)

    data_path = data['data_path']
    fix_meta_data_path = data['meta_data_path']
    if "processed/" + prefix not in data_path:
        data_path = data_path.replace("processed/", "processed/" + prefix + "/")
    if "processed/" + prefix not in fix_meta_data_path:
        fix_meta_data_path = fix_meta_data_path.replace("processed/", "processed/" + prefix + "/")

    data['data_path'] = data_path
    data['meta_data_path'] = fix_meta_data_path

    with open(meta_data_path, "w") as f:
        json.dump(data, f, indent=4)

    return sample_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs")
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    meta = json.load(open(os.path.join(args.input_dir, "meta.json"), "r"))
    sample_ids = meta[0]['sample_ids']

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_sample, args.input_dir, args.prefix, sid): sid
            for sid in sample_ids
        }
        with tqdm(total=len(futures), desc="processing", unit="sample") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)


if __name__ == "__main__":
    main()