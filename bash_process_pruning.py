import os
import subprocess
import sys

OUTPUT_ROOT = "output/dnerf"      
CONFIG_ROOT = "arguments/dnerf"  

ITERATION = 20000
KEEP_RATIO = 1.0
W_DYNAMIC = 1.5
SEARCH_RADIUS = 0.01
D_MAHALANOBIS = 2.0


SKIP_SCENES = [] 

# =====================================================

def run_command(cmd, description):
    print(f"\n[EXEC] {description}...")
    print(f"Command: {cmd}")
    try:

        subprocess.check_call(cmd, shell=True)
        print(f"[SUCCESS] {description} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}.")
        return False

def main():

    if not os.path.exists(OUTPUT_ROOT):
        print(f"Error: Output directory {OUTPUT_ROOT} does not exist.")
        return

    scenes = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    
    print(f"Found {len(scenes)} scenes: {scenes}")
    
    for scene in scenes:
        if scene in SKIP_SCENES:
            print(f"Skipping scene: {scene}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing Scene: {scene}")
        print(f"{'='*60}")

 
        model_path = os.path.join(OUTPUT_ROOT, scene)
        config_path = os.path.join(CONFIG_ROOT, f"{scene}.py")


        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Skipping.")
            continue

      
        cmd_prune = (
            f"{sys.executable} pruning.py "
            f"--model_path {model_path} "
            f"--configs {config_path} "
            f"--iteration {ITERATION} "
            f"--keep_ratio {KEEP_RATIO} "
            f"--w_dynamic {W_DYNAMIC} "
            f"--search_radius {SEARCH_RADIUS} "
            f"--d_mahalanobis {D_MAHALANOBIS}"
        )
        
        if not run_command(cmd_prune, "Pruning"):
            continue 

        cmd_render = (
            f"{sys.executable} render.py "
            f"--model_path {model_path} "
            f"--skip_train "
            f"--configs {config_path} "
            f"--iteration {ITERATION} "
            #f"--pruned"  
        )

        if not run_command(cmd_render, "Rendering"):
            continue


        cmd_metrics = (
            f"{sys.executable} metrics.py "
            f"--model_path {model_path}"
        )

        run_command(cmd_metrics, "Metrics Calculation")

    print("\n\nBatch processing finished!")

if __name__ == "__main__":
    main()