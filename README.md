
## Environmental Setups

Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.

```bash
git clone https://github.com/Enmingzz/4DGaussians.git
cd 4DGaussians
git submodule update --init --recursive
conda create -n gs4d python=3.10 
conda activate gs4d

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

In our environment, we use pytorch=2.9.1+cu128.
GPU: RTX 5090

## Data preparation and train 
Please find the instruction in https://github.com/hustvl/4DGaussians, the data preparation and training sub-section.

## Pruning

You can set your own hyperprameter.

```
python pruning.py \
    --model_path output/dynerf/cut_roasted_beef \
    --configs arguments/dynerf/cut_roasted_beef.py \
    --iteration 14000 \
    --keep_ratio 0.8 \
    --w_dynamic 1.5\
    --search_radius 0.1\
    --d_mahalanobis 2.0
```
## Rendering

Run the following script to render the images for not pruned 4dgs.

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
```
Run the following script to render the images for pruned 4dgs.
```
python render.py \
    --model_path "output/dnerf/bouncingballs/" \
    --skip_train \
    --configs arguments/dnerf/bouncingballs.py \
    --pruned
```
## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```


## Viewer
[Watch me](./docs/viewer_usage.md)
## Scripts

You can use run_batch_dnerf.py to train all data in dnerf dataset.
You can also use batch_process_pruning.py to prune all 4dgs for dnerf dataset.
