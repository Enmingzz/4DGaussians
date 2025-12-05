import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser
from utils.general_utils import safe_state, build_rotation
from scene import Scene, GaussianModel
from tqdm import tqdm
from scipy.spatial import cKDTree
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
try:
    from utils.params_utils import merge_hparams
    import mmcv
    from mmengine.config import Config
except ImportError:
    pass
import torch.nn as nn  

def get_trajectories_cuda(gaussians, frames=20):

    print(f"[Motion] Sampling {frames} frames on CUDA...")
    num_points = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    
    trajectories = torch.empty((num_points, frames * 3), device=device)
    timestamps = torch.linspace(0, 1, frames).to(device)
    
    base_xyz = gaussians.get_xyz.detach()
    base_opacity = gaussians._opacity.detach()
    base_scales = gaussians._scaling.detach()
    base_rotations = gaussians._rotation.detach()
    base_shs = gaussians.get_features.detach()
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(timestamps, desc="Sampling Motion")):
            time_emb = t.view(1, 1).repeat(num_points, 1)
            
            ret = gaussians._deformation(
                base_xyz, 
                base_scales, 
                base_rotations, 
                base_opacity, 
                base_shs, 
                time_emb
            )
            
            d_xyz = ret[0]
            curr_pos = base_xyz + d_xyz
            
            trajectories[:, i*3 : (i+1)*3] = curr_pos
            
    # Normalize for Cosine Similarity
    norm = torch.norm(trajectories, dim=1, keepdim=True)
    trajectories /= (norm + 1e-8)
    return trajectories

def build_cov_cuda(s, r):
    # R from quaternion
    R = build_rotation(r) # [N, 3, 3]
    # S matrix
    S = torch.zeros((s.shape[0], 3, 3), device=s.device)
    S[:, 0, 0] = s[:, 0]; S[:, 1, 1] = s[:, 1]; S[:, 2, 2] = s[:, 2]
    
    L = torch.bmm(R, S)
    Sigma = torch.bmm(L, L.transpose(1, 2))
    return Sigma

def merge_gaussians_cuda(gaussians, args, dataset):
    print(f"Model loaded with {gaussians.get_xyz.shape[0]} Gaussians.")
    
    raw_xyz = gaussians._xyz.detach().cuda()
    raw_opacity = gaussians._opacity.detach().cuda()
    raw_scaling = gaussians._scaling.detach().cuda()
    raw_rotation = gaussians._rotation.detach().cuda()
    raw_features_dc = gaussians._features_dc.detach().cuda()
    raw_features_rest = gaussians._features_rest.detach().cuda()
    
    act_scaling = gaussians.get_scaling.detach().cuda()
    act_opacity = gaussians.get_opacity.detach().cuda()
    
    num_points = raw_xyz.shape[0]
    print(f"Processing {num_points} points on GPU...")
    
    volumes = torch.prod(act_scaling, dim=1)
    sorted_indices = torch.argsort(volumes, descending=True)
    sorted_indices_cpu = sorted_indices.cpu().numpy()
    
    trajectories = None
    if not args.skip_motion:
        trajectories = get_trajectories_cuda(gaussians, frames=args.sample_frames)
    
    print("Pre-computing Covariances...")
    all_sigma = build_cov_cuda(act_scaling, raw_rotation)
    all_inv_sigma = torch.linalg.inv(all_sigma + 1e-6 * torch.eye(3, device="cuda"))
    
    print("Building KDTree...")
    xyz_cpu = raw_xyz.cpu().numpy()
    tree = cKDTree(xyz_cpu)
    
    merged_mask = torch.zeros(num_points, dtype=torch.bool, device="cuda")
    
    new_xyz_list = []
    new_s_list = []   
    new_r_list = []   
    new_o_list = []    
    new_sh_dc_list = []
    new_sh_rest_list = []
    
    merge_count = 0
    removed_count = 0
    
    print("Start Merging...")
    pbar = tqdm(total=num_points)
    
    def inverse_opacity(o):
        # Inverse Sigmoid: log(p / (1-p))
        o = torch.clamp(o, min=1e-6, max=1.0 - 1e-6)
        return torch.log(o / (1 - o))

    def append_result(xyz_val, scale_raw, rot_raw, op_raw, sh_dc, sh_rest):
        new_xyz_list.append(xyz_val)
        new_s_list.append(scale_raw)
        new_r_list.append(rot_raw)
        new_o_list.append(op_raw)
        new_sh_dc_list.append(sh_dc)
        new_sh_rest_list.append(sh_rest)

    for i in sorted_indices_cpu:
        if merged_mask[i]:
            pbar.update(1); continue
            
        # Broad Phase
        neighbors_idx = tree.query_ball_point(xyz_cpu[i], r=args.search_radius)
        
        # Case 1: No Neighbors -> Keep Original
        if len(neighbors_idx) <= 1:
            merged_mask[i] = True
            append_result(raw_xyz[i], raw_scaling[i], raw_rotation[i], raw_opacity[i], raw_features_dc[i], raw_features_rest[i])
            pbar.update(1); continue
            
        cand_indices = torch.tensor(neighbors_idx, device="cuda", dtype=torch.long)
        valid_mask = ~merged_mask[cand_indices]
        cand_indices = cand_indices[valid_mask]
        
        # Case 2: Neighbors gone -> Keep Original
        if len(cand_indices) <= 1:
            merged_mask[i] = True
            append_result(raw_xyz[i], raw_scaling[i], raw_rotation[i], raw_opacity[i], raw_features_dc[i], raw_features_rest[i])
            pbar.update(1); continue

        # --- Narrow Phase ---
        diff = raw_xyz[cand_indices] - raw_xyz[i]
        term = torch.matmul(diff, all_inv_sigma[i]) 
        dist_sq = torch.sum(term * diff, dim=1) 
        mask_spatial = dist_sq < (args.d_mahalanobis ** 2)
        
        # SH Sim (using Raw DC is fine as it's linear-ish space)
        sh_i_flat = raw_features_dc[i].flatten()
        sh_cand_flat = raw_features_dc[cand_indices].reshape(len(cand_indices), -1)
        sh_sim = torch.nn.functional.cosine_similarity(sh_i_flat.unsqueeze(0), sh_cand_flat)
        mask_sh = sh_sim > args.rho_sh
        
        if trajectories is not None:
            traj_i = trajectories[i]
            traj_cand = trajectories[cand_indices]
            motion_sim = torch.sum(traj_i * traj_cand, dim=1)
            mask_motion = motion_sim > args.tau_motion
        else:
            mask_motion = torch.ones_like(mask_sh, dtype=torch.bool)
            
        final_mask = mask_spatial & mask_sh & mask_motion
        final_cluster_indices = cand_indices[final_mask]
        
        merged_mask[final_cluster_indices] = True
        
        if len(final_cluster_indices) == 1:
            idx = final_cluster_indices[0]
            append_result(raw_xyz[idx], raw_scaling[idx], raw_rotation[idx], raw_opacity[idx], raw_features_dc[idx], raw_features_rest[idx])
        else:
            # --- Merging Logic ---
            # 1. Weights using Activated Opacity
            cluster_ops = act_opacity[final_cluster_indices]
            weights = cluster_ops / (cluster_ops.sum() + 1e-8)
            
            # 2. Dominant Point (for Shape)
            best_local_idx = torch.argmax(weights).item()
            dominant_idx = final_cluster_indices[best_local_idx]
            
            # 3. Weighted Position (XYZ)
            cluster_xyz = raw_xyz[final_cluster_indices]
            mu_q = torch.sum(weights * cluster_xyz, dim=0)
            
            # 4. Weighted SH
            sh_dc_q = torch.sum(weights.view(-1, 1, 1) * raw_features_dc[final_cluster_indices], dim=0)
            sh_rest_q = torch.sum(weights.view(-1, 1, 1) * raw_features_rest[final_cluster_indices], dim=0)
            
            # 5. Sum Opacity (Activated -> Inverse -> Raw)
            sum_op = torch.clamp(torch.sum(cluster_ops, dim=0), max=0.999)
            raw_op_new = inverse_opacity(sum_op)
            
            append_result(mu_q, raw_scaling[dominant_idx], raw_rotation[dominant_idx], raw_op_new, sh_dc_q, sh_rest_q)
            
            merge_count += 1
            removed_count += (len(final_cluster_indices) - 1)
        pbar.update(1)
        
    print(f"Merging Done. Final: {len(new_xyz_list)}. Removed: {removed_count}")
    
    # Stack
    f_xyz = torch.stack(new_xyz_list)
    f_scaling = torch.stack(new_s_list)
    f_rotation = torch.stack(new_r_list)
    f_opacity = torch.stack(new_o_list)
    f_sh_dc = torch.stack(new_sh_dc_list)
    f_sh_rest = torch.stack(new_sh_rest_list)
    
    # Update Parameters
    gaussians._xyz = nn.Parameter(f_xyz)
    gaussians._scaling = nn.Parameter(f_scaling)
    gaussians._rotation = nn.Parameter(f_rotation)
    gaussians._opacity = nn.Parameter(f_opacity)
    gaussians._features_dc = nn.Parameter(f_sh_dc)
    gaussians._features_rest = nn.Parameter(f_sh_rest)
    
    # Rebuild index
    if hasattr(gaussians, '_deformation_table'):
        gaussians._deformation_table = torch.arange(f_xyz.shape[0], device="cuda", dtype=torch.float32)

    # Save
    save_dir = os.path.join(dataset.model_path, "point_cloud/iteration_"+str(args.iteration))
    ply_path = os.path.join(save_dir, f"pruned{args.iteration}.ply")
    print(f"Saving to {ply_path}...")
    gaussians.save_ply(ply_path)


def manual_prune(gaussians, keep_mask):

    print("Executing manual pruning (bypassing optimizer)...")
    
    def apply_mask(tensor, mask):
        if tensor is None:
            return None
        if tensor.shape[0] != mask.shape[0]:
            return tensor
        if tensor.device != mask.device:
            return tensor[mask.to(tensor.device)]
        return tensor[mask]
    # ---------------------------------

    gaussians._xyz = nn.Parameter(apply_mask(gaussians._xyz, keep_mask))
    gaussians._features_dc = nn.Parameter(apply_mask(gaussians._features_dc, keep_mask))
    gaussians._features_rest = nn.Parameter(apply_mask(gaussians._features_rest, keep_mask))
    gaussians._opacity = nn.Parameter(apply_mask(gaussians._opacity, keep_mask))
    gaussians._scaling = nn.Parameter(apply_mask(gaussians._scaling, keep_mask))
    gaussians._rotation = nn.Parameter(apply_mask(gaussians._rotation, keep_mask))

    if hasattr(gaussians, '_deformation_table'):
        gaussians._deformation_table = apply_mask(gaussians._deformation_table, keep_mask)
    
    if hasattr(gaussians, 'xyz_gradient_accum'):
        gaussians.xyz_gradient_accum = apply_mask(gaussians.xyz_gradient_accum, keep_mask)
        
    if hasattr(gaussians, 'denom'):
        gaussians.denom = apply_mask(gaussians.denom, keep_mask)
        
    if hasattr(gaussians, 'max_radii2D'):
        gaussians.max_radii2D = apply_mask(gaussians.max_radii2D, keep_mask)
        
    print(f"Pruning done. New model size: {gaussians.get_xyz.shape[0]}")

def normalize(x):
    min_val = x.min()
    max_val = x.max()
    if max_val - min_val < 1e-8:
        return torch.zeros_like(x)
    return (x - min_val) / (max_val - min_val)

def compute_anisotropy(scaling):
    s_max = torch.max(scaling, dim=1).values
    s_min = torch.min(scaling, dim=1).values
    return s_max / (s_min + 1e-6)

def compute_dynamic_stats_hustvl(gaussians, total_frames=30):
    print(f"[HUSTVL] Computing dynamic statistics (sampling {total_frames} frames)...")
    num_points = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    
    accum_velocity_sq = torch.zeros(num_points, device=device)
    accum_acceleration_norm = torch.zeros(num_points, device=device)
    accum_opacity_sum = torch.zeros(num_points, device=device)
    accum_opacity_sq_sum = torch.zeros(num_points, device=device)
    
    prev_xyz = None
    prev_prev_xyz = None
    
    timestamps = torch.linspace(0, 1, total_frames).to(device)
    
    base_xyz = gaussians.get_xyz.detach()
    base_opacity = gaussians._opacity.detach()
    base_scales = gaussians._scaling.detach()
    base_rotations = gaussians._rotation.detach()
    base_shs = gaussians.get_features.detach()
    
    with torch.no_grad():
        for t in tqdm(timestamps, desc="Walking through time"):
            time_emb = t.view(1, 1).repeat(num_points, 1)
            
            ret = gaussians._deformation(
                base_xyz, 
                base_scales, 
                base_rotations, 
                base_opacity, 
                base_shs, 
                time_emb
            )
            
            current_xyz = ret[0]
            current_opacity = ret[3] 
            current_opacity = gaussians.opacity_activation(current_opacity)

            accum_opacity_sum += current_opacity.squeeze()
            accum_opacity_sq_sum += (current_opacity.squeeze() ** 2)
            
            if prev_xyz is not None:
                delta_x = current_xyz - prev_xyz
                velocity_sq = torch.sum(delta_x ** 2, dim=1)
                accum_velocity_sq += velocity_sq
                
                if prev_prev_xyz is not None:
                    acc_vec = current_xyz - 2 * prev_xyz + prev_prev_xyz
                    acc_norm = torch.norm(acc_vec, dim=1)
                    accum_acceleration_norm += acc_norm
            
            prev_prev_xyz = prev_xyz
            prev_xyz = current_xyz
            
    D_i = torch.sqrt(accum_velocity_sq / (total_frames - 1 + 1e-6))
    if total_frames > 2:
        kappa_i = accum_acceleration_norm / (total_frames - 2 + 1e-6)
    else:
        kappa_i = torch.zeros_like(D_i)
        
    mean_opacity = accum_opacity_sum / total_frames
    mean_sq_opacity = accum_opacity_sq_sum / total_frames
    V_i = torch.clamp(mean_sq_opacity - mean_opacity**2, min=0)
    
    return D_i, kappa_i, V_i
def prune_model(dataset, hyper, args):
    print(f"Loading model from {dataset.model_path}...")

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    
    # --- Stage 1: Calculating Scores ---
    print("\n[Stage 1] Calculating Importance Scores...")
    print("zzzzzzzzzz", gaussians.get_xyz[0:5])
    # Static Score
    print(dataset.sh_degree)
    vol_proxy = torch.prod(gaussians.get_scaling, dim=1)
    C_i = gaussians.get_opacity.squeeze() * vol_proxy
    E_i = compute_anisotropy(gaussians.get_scaling)
    S_static = normalize(C_i) + args.lambda_weight * normalize(E_i)
    print(S_static[0:10])
    
    # Dynamic Score
    if args.skip_dynamic:
        S_dynamic = torch.zeros_like(S_static)
    else:
        D_i, kappa_i, V_i = compute_dynamic_stats_hustvl(gaussians, total_frames=args.sample_frames)
        S_dynamic = normalize(D_i) + args.mu_weight * normalize(kappa_i) + args.eta_weight * normalize(V_i)
        
    total_score = args.w_static * S_static + args.w_dynamic * S_dynamic
    
    print(total_score[0:10])
    # --- Stage 2: Pruning ---
    n_init = gaussians.get_xyz.shape[0]
    print(f"\n[Stage 2] Pruning (Initial: {n_init})...")
    
    if args.method == 'percentile':
        threshold = torch.quantile(total_score, 1 - args.keep_ratio)
        mask = total_score >= threshold
        print(f"Method: Top {args.keep_ratio*100:.1f}% | Score Threshold: {threshold:.4f}")
    else:
        mask = total_score >= args.threshold
        print(f"Method: Absolute Threshold | Score Threshold: {args.threshold}")

    n_keep = mask.sum().item()
    print(f"Removing {n_init - n_keep} points. Remaining: {n_keep}")
    
    manual_prune(gaussians, mask)
    print(gaussians.get_xyz.shape[0], "Gaussians remain after pruning.")
    
    # Saving
    save_dir = os.path.join(dataset.model_path, "point_cloud/iteration_"+str(args.iteration))
    os.makedirs(save_dir, exist_ok=True)
    
    ply_path = os.path.join(save_dir, f"pruned{args.iteration}.ply")
    gaussians.save_ply(ply_path)

    merge_gaussians_cuda(gaussians, args, dataset)
    print("Pruned and merged model saved.")


if __name__ == "__main__":
    parser = ArgumentParser(description="HUSTVL 4DGS Pruning Script")
    
    lp = ModelParams(parser, sentinel=True)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--suffix", default="_fine", type=str)
    parser.add_argument("--configs", type=str, default="", help="Path to MMCV config file (CRITICAL)")
    
    parser.add_argument("--w_static", default=1.0, type=float)
    parser.add_argument("--w_dynamic", default=1.0, type=float)
    parser.add_argument("--lambda_weight", default=0.5, type=float)
    parser.add_argument("--mu_weight", default=0.5, type=float)
    parser.add_argument("--eta_weight", default=0.5, type=float)
    
    parser.add_argument("--method", default='percentile', choices=['percentile', 'absolute'])
    parser.add_argument("--keep_ratio", default=0.6, type=float, help="Ratio of points to KEEP")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--sample_frames", default=20, type=int)
    parser.add_argument("--skip_dynamic", action="store_true")
    parser.add_argument("--skip_merge", action="store_true", help="Skip the merging step")

    parser.add_argument("--search_radius", default=0.02, type=float)
    parser.add_argument("--d_mahalanobis", default=2.0, type=float)
    parser.add_argument("--rho_sh", default=0.9, type=float)
    parser.add_argument("--tau_motion", default=0.8, type=float)
    parser.add_argument("--skip_motion", action="store_true")
    
    args = get_combined_args(parser)
    if args.configs:
        config = Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    print(f"Optimizing: {args.model_path}")
    #safe_state(False)
    
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    dataset.model_path = args.model_path # Ensure path is propagated

    print(dataset)
    
    prune_model(dataset, hyper, args)