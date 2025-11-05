"""Setup RIO10 dataset.

See: https://waldjohannau.github.io/RIO10/

Script adapted from the official download script (which you only get access to after filling out the form).
Redacted the download URL to not circumvent the form.
"""

# Downloads RIO10 public data release

# The data is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.
# RIO10 consists of a subset of 10 scenes from 3RScan.
# Each scene consists of separate folders for every scan where scene<scene_id>_01 is always the training sequence.
# Visit our project website for more information and check out our toolkit on GitHub: github.com/WaldJohannaU/RIO10
#
# Script usage:
# - To download the entire RIO10 release: download.py -o [directory in which to download]
# - To download a specific scene (e.g., 1-10): download.py -o [directory in which to download] --id 1
# - To download the kapture files (e.g., 1-10): download.py -o [directory in which to download] --id 1 --type=kapture

import argparse
import json
import os
import pathlib
import tempfile
import urllib.request as urllib

import numpy as np
import ruamel.yaml as _yaml

BASE_URL = "TODO"  # Retrieve by following the instructions at https://github.com/WaldJohannaU/RIO10?tab=readme-ov-file#download
DATA_URL = BASE_URL + "Dataset/"
TOS_URL = BASE_URL + "RIO10TOU.pdf"
FILETYPES = ["seq", "models", "semantics", "kapture"]

raw_folder = "rio10_raw"
ace_folder = "rio10"

yaml = _yaml.YAML()


def download_release(out_dir, file_types):
    print("Downloading RIO10 release to " + out_dir + "...")
    for scene_id in range(1, 11):  # 1-10
        download_scan(scene_id, out_dir, file_types)
    print("Downloaded RIO10 release.")


def download_file(url, out_file):
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print("\t" + url + " > " + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, "w")
        f.close()
        urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print("WARNING: skipping download of existing file " + out_file)


def download_scan(scene_id, out_dir, file_types):
    scene_str = "scene" + str(scene_id).zfill(2)
    out_scene_dir = os.path.join(out_dir, scene_str)
    print("Downloading RIO10", scene_id, "...", out_dir, file_types, out_scene_dir)
    if BASE_URL == "TODO":
        print("ERROR: Download URL is not set. Please follow the instructions at "
              "https://github.com/WaldJohannaU/RIO10?tab=readme-ov-file#download to retrieve the download URL and edit "
              "this file accordingly.")
        exit(1)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        if ft == "kapture":
            for file_type in ["mapping", "testing", "validation"]:
                url = DATA_URL + "/kapture/RIO10_scene" + str(scene_id).zfill(2) + "_" + file_type + ".tar.gz"
                out_file = out_scene_dir + "/RIO10_scene" + str(scene_id).zfill(2) + "_" + file_type + ".tar.gz"
                download_file(url, out_file)
        else:
            url = DATA_URL + "/" + ft + str(scene_id).zfill(2) + ".zip"
            out_file = out_scene_dir + "/" + ft + str(scene_id).zfill(2) + ".zip"
            download_file(url, out_file)


def get_filename(out_dir, scene_id):
    return os.path.join(out_dir, "scene" + str(scene_id).zfill(2))


def main():
    parser = argparse.ArgumentParser(description="Downloads RIO10 public data release.")
    parser.add_argument("--id", type=int, help="specific scene id to download [1-10]")
    parser.add_argument("--type", help="specific file type to download")
    args = parser.parse_args()

    print("By pressing any key to continue you confirm that you have agreed to the RIO10 terms of use as described at:")
    print(TOS_URL)
    print("***")
    print("Press any key to continue, or CTRL-C to exit.")
    key = input("")

    file_types = ["seq", "models", "semantics"]

    if args.type:  # download file type
        file_type = args.type
        if file_type not in FILETYPES:
            print("ERROR: Invalid file type: " + file_type)
            return
        file_types = [file_type]
    if args.id:  # download single scene
        scene_id = int(args.id)
        if scene_id > 10 and scene_id <= 0:
            print("ERROR: Invalid scan id: " + scene_id)
        elif scene_id <= 10 and scene_id > 0:
            download_scan(scene_id, args.out_dir, file_types)
    else:  # download entire release
        if len(file_types) == len(FILETYPES):
            print("Downloading the entire RIO10 release.")
        else:
            print("Downloading all RIO10 scans of type " + file_types[0])
        print(
            "Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download."
        )
        print("***")
        print("Press any key to continue, or CTRL-C to exit.")
        key = input("")

        download_release(raw_folder, file_types)

    # convert to ACE format
    with urllib.urlopen(
        "https://raw.githubusercontent.com/WaldJohannaU/RIO10/refs/heads/master/data/metadata.json"
    ) as url:
        splits = json.load(url)

    # with urllib.urlopen(
    #     "https://github.com/WaldJohannaU/RIO10/blob/master/data/intrinsics.txt"
    # ) as url:
    #     intrinsics = url.read().decode("utf-8")

    raw_scene_dirs = list(pathlib.Path(raw_folder).glob("*"))
    for raw_scene_dir in raw_scene_dirs:
        scene_name = raw_scene_dir.stem
        scene_id = scene_name[-2:]
        zip_file = raw_scene_dir / f"seq{scene_id}.zip"
        seqs_dir = raw_scene_dir / f"seq{scene_id}"
        if not seqs_dir.exists():
            os.system(f"unzip {zip_file} -d {raw_scene_dir}")
        scene_splits = next(split for split in splits if split["train"].startswith(f"seq{scene_id}"))
        for split_name, seqs in scene_splits.items():
            if split_name == "test":
                split_name = "hidden_test"
            if split_name == "val":
                split_name = "test"
            if isinstance(seqs, str):
                seqs = [seqs]

            out_scene_dir = pathlib.Path(ace_folder) / scene_name
            out_split_dir = out_scene_dir / split_name
            out_split_dir.mkdir(parents=True, exist_ok=True)

            (out_split_dir / "rgb").mkdir(parents=True, exist_ok=True)
            (out_split_dir / "depth").mkdir(parents=True, exist_ok=True)
            (out_split_dir / "poses").mkdir(parents=True, exist_ok=True)

            for seq in seqs:
                camera_yaml = seqs_dir / seq / "camera.yaml"
                with open(camera_yaml, "r") as file:
                    camera_data = yaml.load(file)
                fx, fy, cx, cy = camera_data["camera_intrinsics"]["model"]
                np.savetxt(
                    out_split_dir / "calibration.txt",
                    np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                )

                for rgb_file in (seqs_dir / seq).glob("*.color.jpg"):
                    out_path = out_split_dir / "rgb" / (seq + rgb_file.name)
                    out_path.symlink_to(os.path.relpath(rgb_file.absolute(), out_path.parent))
                for depth_file in (seqs_dir / seq).glob("*.depth.png"):
                    out_path = out_split_dir / "depth" / (seq + depth_file.name)
                    out_path.symlink_to(os.path.relpath(depth_file.absolute(), out_path.parent))
                for pose_file in (seqs_dir / seq).glob("*.pose.txt"):
                    out_path = out_split_dir / "poses" / (seq + pose_file.name)
                    out_path.symlink_to(os.path.relpath(pose_file.absolute(), out_path.parent))


if __name__ == "__main__":
    main()
