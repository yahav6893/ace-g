# ACE-G: Improving Scene Coordinate Regression Through Query Pre-Training

<div align="center">

[Leonard Bruns](https://roym899.github.io/), [Axel Barroso-Laguna](https://scholar.google.com/citations?user=m_SPRGUAAAAJ), [Tommaso Cavallari](https://scholar.google.it/citations?user=r7osSm0AAAAJ), [Áron Monszpart](https://amonszpart.github.io/), [Sowmya Munukutla](https://scholar.google.com/citations?user=l-zRzDEAAAAJ), [Victor Adrian Prisacariu](https://www.robots.ox.ac.uk/~victor/), [Eric Brachmann](https://ebrach.github.io/)

[![ACE-G](./resources/teaser.gif)](https://nianticspatial.github.io/ace-g/)

</div>

This repository contains the code for "ACE-G: Improving Scene Coordinate Regression Through Query Pre-Training". For illustrations of the method and additional visualizations check out the [project page](https://nianticspatial.github.io/ace-g/) and the [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Bruns_ACE-G_Improving_Generalization_of_Scene_Coordinate_Regression_Through_Query_Pre-Training_ICCV_2025_paper.html). We provide the code to find the map code for novel scenes, relocalize images within, visualize the results, and reproduce the results from the paper. Currently, there are no plans to release the code for pre-training, but we do provide the pre-trained model used in the paper.

## 0 - Quick Start
We provide a [pixi environment](https://pixi.sh/latest/installation/) to quickly run and visualize an example of mapping and relocalization with a single command. Alternatively, you can setup the environment manually by following the instructions [below](#1---setup).

To run the example, first install pixi, and then use the following command:
```bash
pixi run example
```
which will setup the Indoor-6 dataset, install the dependencies, and run a mapping and relocalization experiment with visualization enabled (all visualizations are based on Rerun, see [below](#rerun-visualization) for guidance on how to use it).

By default the example requires 21GB of VRAM. To run it with less VRAM, there are two main parameters that reduce memory usage: first, `--max_buffer_size` can be reduced from its default `4000000`; second, `--batch_size` can be reduced from its default `40960`. Note that results might be slighty worse than reported in the paper when these are changed. Increasing the number of iterations and training for longer might make up for some of this (increase `--num_iterations` from the default `1000`).

## 1 - Setup
When using pixi and the example worked, you are already done with the environment setup. Use `pixi shell` to enter the pixi environment. Otherwise, you need to install `libopencv` and then run the following command to install the dependencies from pypi in an environment of your choice.
```bash
pip install -e .
```
Then download our pre-trained model using
```
wget https://storage.googleapis.com/niantic-lon-static/research/ace-g/ace_g_pretrained.pt
```

## 1.1 - Setup datasets
The dataset setup is mainly automated. We provide setup scripts in the `datasets` directory for the following datasets: 7Scenes, 12Scenes, Cambridge, Indoor-6, and RIO-10. To setup a dataset, change to the `datasets` directory and run the setup script for the dataset of interest. For example, to setup the Indoor-6 dataset, run the following command:
```bash
cd datasets
python setup_indoor6.py
```
This will download the dataset and bring it into the correct format to be used with this codebase. Some datasets require filling out a form to download the dataset. In these cases, you will be prompted to do so when running the script. Refer to the respective setup scripts for more details.

#### 1.1.1 - Visualize datasets
Prior to mapping when running on a new dataset, it is useful to visualize the dataset to ensure that the data is in the expected format and that the camera poses are correct. To do this, run the following command:
```bash
python -m ace_g.vis_dataset \
  --rgb_files "path_to_rgb_files/*.jpg" \
  --pose_files "path_to_pose_files/*.txt" \
  --depth_files "path_to_depth_files/*.png" \
  --subsample_factor 14 \
  --use_color True
```
The depth files are optional, but when available they offer an easy way to visually assess the correctness of the dataset.

## 2 - Mapping
To run mapping for a single scene using ACE-G's default configuration, run the following command:

```bash
python -m ace_g.train_single_scene \
  --config "ace_g_5min.yaml" \
  --dataset.rgb_files "path_to_rgb_files/*.jpg" \
  --dataset.pose_files "path_to_pose_files/*.txt"
```

This will train the model and save the results in the `outputs` directory. Two important files are produced: `{session_name}_map.yaml` and `{session_name}_map.pt`. The first contains the mapping configuration (and also points to the second file) and can be used to reproduce this particular mapping run. The second file contains the optimized map codes for the scene.

Our code also allows to run ACE, its DINOv2 variant, and ACE-G's 25 minute configuration by using `--config ace.yaml`, `--config ace-dinov2.yaml`, and `--config ace_g_25min.yaml`, respectively.

### 2.1 - Visualize mapping
Add `--use_rerun True` and `--rerun_spawn True` to the mapping command to visualize the losses and 3D coordinates using Rerun.

## 3 - Relocalization
To run relocalization on a set of query images using the previously learned map codes, run the following command:
```bash
python -m ace_g.register_images \
  --config "path_to_mapping_results/{session_name}_map.yaml" \
  --dataset.rgb_files 'path_to_query_images/*.jpg'
```
This will produce a file `{session_name}_registered_poses.txt` containing the estimated poses for the query images.

### 3.1 - Visualize relocalization
To visualize the relocalization use the following command:
```bash
python -m ace_g.vis_localization \
  --config "path_to_mapping_results/{session_name}_map.yaml" \
  --dataset.rgb_files 'path_to_query_images/*.jpg' \
  --dataset.pose_files 'path_to_gt_poses/*.txt'
```

If the dataset follows the same structure as the included datasets, you can use `--switch_sequence True` or `--switch_sequence 'other_sequence_name'` to switch from the mapping sequence to another sequence (this is often more convenient than providing explicit file paths as shown above).

## 4 - Reproduce Results
Check the file `run_eval.sh` for the commands to reproduce our main results. In addition to the previous commands it includes the call to evaluate the poses after relocalization. The final per-scene metrics can be found in the `outputs/{session_name}_eval.txt` and `outputs/{session_name}_eval.yaml` files.

## Rerun Visualization
The visualizations are based on [Rerun](https://rerun.io/), which can be used in [a few different operating modes](https://rerun.io/docs/reference/sdk/operating-modes) depending on your exact setup. By default, this codebase assumes that you are using a standard desktop machine, in which case you have nothing to change and the viewer will show up automatically. We configured the default views so it is hopefully easy to get started. In all cases you will see a 3D view that you can interact with and a timeline at the bottom, which can be used to inspect the data over time.

If you use a headless machine that you ssh into, your easieset option is to use `--rerun_spawn False` (see the `run_example.sh` file) and a reverse tunnel with the default port 9876 (e.g., `ssh -R 9876:127.0.0.1:9876 ...`). Then you need to install a matching Rerun version on your machine (e.g., `pip install rerun-sdk==0.26.2`) and start it locally with `rerun`. Once you run the code on the remote with this setup, it should automatically stream the data to your local viewer.

## Citation
If you find this work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{bruns2025aceg,
  title={{ACE-G}: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training},
  author={Bruns, Leonard and Barroso-Laguna, Axel and Cavallari, Tommaso and Monszpart, {\'{A}}ron and Munukutla, Sowmya and Prisacariu, Victor and Brachmann, Eric},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
