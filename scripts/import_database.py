import argparse
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from supabase import create_client

_WORKER_DB = None


def _init_worker(database_url: str, database_key: str) -> None:
    global _WORKER_DB
    _WORKER_DB = create_client(database_url, database_key)


def _update_sample(
    input_dir: str,
    database_table: str,
    sample_id: str,
    render_sample_ids: set[str],
) -> str:
    if _WORKER_DB is None:
        raise RuntimeError("worker database client not initialized")

    meta_data_path = os.path.join(input_dir, "meta_data", sample_id + ".json")
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)

    data_path = meta_data['data_path']
    load_meta_data_path = meta_data['meta_data_path']
    path = meta_data['video_path']
    render_video_path = os.path.join(
        os.path.dirname(os.path.dirname(load_meta_data_path)),
        "render", sample_id + ".mp4",
    )

    result = {
        'agilex_ik_result_data_path': data_path,
        'agilex_ik_result_meta_data_path': load_meta_data_path,
        'agilex_rendered_video_path': render_video_path if sample_id in render_sample_ids else None,
    }

    _WORKER_DB.table(database_table).update(result).eq("path", path).execute()
    return sample_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs")
    parser.add_argument("--database_url", type=str, default="http://192.168.3.208:54321")
    parser.add_argument("--database_key", type=str, default="sb_publishable_ACJWlzQHlZjBrEguHvfOxg_3BJgxAaH")
    parser.add_argument("--database_table", type=str, default="egocentric_dataset_clips")
    parser.add_argument("--num_workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    meta_path = os.path.join(args.input_dir, "meta.json")
    all_sample_ids = json.load(open(meta_path, "r"))[0]['sample_ids']
    render_meta_path = os.path.join(args.input_dir, "render_meta.json")
    render_sample_ids = set(json.load(open(render_meta_path, "r")))

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_worker,
        initargs=(args.database_url, args.database_key),
    ) as executor:
        futures = {
            executor.submit(
                _update_sample,
                args.input_dir,
                args.database_table,
                sid,
                render_sample_ids,
            ): sid
            for sid in all_sample_ids
        }
        failed = []
        with tqdm(total=len(futures), desc="importing", unit="sample") as pbar:
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((sid, str(e)))
                pbar.update(1)

    if failed:
        print(f"\n{len(failed)} samples failed:")
        for sid, err in failed[:20]:
            print(f"  {sid}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()