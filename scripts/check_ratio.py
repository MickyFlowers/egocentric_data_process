import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    meta_file = os.path.join(args.input_dir, "meta.json")
    render_meta_file = os.path.join(args.input_dir, "render_meta.json")
    meta = json.load(open(meta_file, "r"))
    render_meta = json.load(open(render_meta_file, "r"))
    ratio = len(render_meta) / len(meta[0]['sample_ids'])
    print("success ratio: ", ratio)

if __name__ == "__main__":
    main()

    