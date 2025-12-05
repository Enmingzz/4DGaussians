import os
import sys
import subprocess
import json
import glob
import re


DATA_ROOT = "data/dnerf"  
OUTPUT_ROOT = "output/dnerf"   
CONFIG_ROOT = "arguments/dnerf" 
PYTHON_CMD = sys.executable    


SKIP_SCENES = [] 
# ===========================================

def get_gaussian_count(model_path, iteration=30000):
    """
    通过读取 PLY 文件头来获取高斯球数量，避免依赖额外库。
    """
 
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    
    if not os.path.exists(ply_path):
        return 0

    try:
        with open(ply_path, 'rb') as f:
 
            header = b""
            while b"end_header" not in header:
                header += f.read(1024)
            
 
            match = re.search(rb"element vertex (\d+)", header)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error reading PLY file {ply_path}: {e}")
    
    return 0

def get_psnr_from_json(model_path):
    """
    尝试从 metrics.py 生成的 JSON 结果中读取 PSNR
    """

    json_path = os.path.join(model_path, "results.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if 'ours_30000' in data and 'PSNR' in data['ours_30000']:
                    return float(data['ours_30000']['PSNR'])
                elif 'PSNR' in data:
                    return float(data['PSNR'])
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
    
    return 0.0

def run_batch():

    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root '{DATA_ROOT}' not found.")
        return

    
    scenes = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    scenes.sort()
    
    print(f"Found {len(scenes)} scenes: {scenes}")
    
    results = []

    for scene in scenes:
        if scene in SKIP_SCENES:
            print(f"Skipping {scene}...")
            continue

        print(f"\n{'='*20} Processing Scene: {scene} {'='*20}")
        

        scene_data_path = os.path.join(DATA_ROOT, scene) # data/dnerf/bouncingballs
        expname = f"dnerf/{scene}"                       # dnerf/bouncingballs
        model_path = os.path.join(OUTPUT_ROOT, scene)    # output/dnerf/bouncingballs/
        config_path = os.path.join(CONFIG_ROOT, f"{scene}.py") # arguments/dnerf/bouncingballs.py
        
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Skipping {scene}.")
            continue

        # --- Step 1: Training ---
        # python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py
        cmd_train = [
            PYTHON_CMD, "train.py",
            "-s", scene_data_path,
            "--port", "6017",
            "--expname", expname,
            "--configs", config_path
        ]
        
        # --- Step 2: Rendering ---
        # python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py 
        cmd_render = [
            PYTHON_CMD, "render.py",
            "--model_path", model_path,
            "--skip_train",
            "--configs", config_path
        ]
        
        # --- Step 3: Metrics ---
        # python metrics.py --model_path "output/dnerf/bouncingballs/" 
        cmd_metrics = [
            PYTHON_CMD, "metrics.py",
            "--model_path", model_path
        ]



    print("\n\n" + "="*40)
    print("FINAL BATCH REPORT")
    print("="*40)
    print(f"{'Scene':<20} | {'Gaussians':<15} | {'PSNR':<10}")
    print("-" * 50)
    
    total_gaussians = 0
    total_psnr = 0.0
    valid_count = 0
    
    for res in results:
        print(f"{res['scene']:<20} | {res['gaussians']:<15} | {res['psnr']:<10.2f}")
        if res['psnr'] > 0: 
            total_gaussians += res['gaussians']
            total_psnr += res['psnr']
            valid_count += 1
            
    print("-" * 50)
    if valid_count > 0:
        avg_gaussians = int(total_gaussians / valid_count)
        avg_psnr = total_psnr / valid_count
        print(f"{'AVERAGE':<20} | {avg_gaussians:<15} | {avg_psnr:<10.2f}")
    else:
        print("No valid results found.")
    print("="*40)

if __name__ == "__main__":
    run_batch()