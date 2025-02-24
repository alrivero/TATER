import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import os
import torchvision.transforms as T
from spectre import SPECTRE
#from src.FLAME.FLAME import FLAME
from renderer.renderer import Renderer
from VOCASET_dataset import get_datasets_VOCASET
import numpy as np
import torch.nn.functional as F
import mediapipe as mp
import cv2
import ffmpeg
import soundfile as sf
import torchvision.utils as vutils
#from datasets.VOCASET_dataset import get_datasets_VOCASET
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss.point_mesh_distance import point_mesh_face_distance
from mixed_dataset_sampler import MixedDatasetBatchSampler
import pandas as pd
from pathlib import Path

SEG_LEN = 24
OVERLAP = 4
FACE_CROP = True
ICP_BATCH_SIZE = 8  # Adjust based on VRAM availability


def parse_args():
    conf = OmegaConf.load(sys.argv[1])
    OmegaConf.set_struct(conf, True)
    sys.argv = [sys.argv[0]]# + sys.argv[2:]  # Remove config file from args
    conf.merge_with_cli()
    return conf


def force_model_to_device(model, device):
    """
    Moves all parameters and buffers of the model to a specific device.
    """
    device = torch.device(device)

    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)

    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)

    model.to(device)
    print(f"✅ Model moved to {device}")



def detect_face_and_crop(image_batch, detector, target_size=224, padding_value=0, scale_factor=1.7):
    """
    Detects faces using MediaPipe Face Detection and applies **tight, square cropping centered on the face**.
    If the crop is **out of bounds**, it is **padded** to maintain a centered, square image.

    **Scaling is performed by first centering the bounding box at the origin, scaling, and then shifting it back.**

    Args:
        image_batch (torch.Tensor): Batch of images **(B, 3, H, W)** or a single image **(3, H, W)** normalized to [0,1].
        detector: MediaPipe FaceDetection model (must be instantiated **outside** the function).
        target_size (int): Output size of the cropped face.
        padding_value (float): Pixel fill value for out-of-bounds regions (default: `0` for black padding).
        scale_factor (float): Multiplier for expanding/shrinking the detected face crop.

    Returns:
        cropped_images (torch.Tensor): Cropped faces resized to (B, 3, target_size, target_size) or (3, target_size, target_size) if input was a single image.
        bboxes (list of tuples): [(x_min, y_min, x_max, y_max)] for each image (None if no face detected).
        centers (list of tuples): [(x_center, y_center)] for each image (None if no face detected).
    """
    if image_batch.dim() == 3:  # Convert single image to batch (1, 3, H, W)
        image_batch = image_batch.unsqueeze(0)

    B, _, H, W = image_batch.shape
    cropped_images, bboxes, centers, paddings = [], [], [], []

    for i in range(B):
        image_np = (image_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Convert to NumPy

        results = detector.process(image_np)  # Call process directly

        if results.detections:
            # **Get first detected face bounding box**
            bbox = results.detections[0].location_data.relative_bounding_box
            cx, cy = bbox.xmin + bbox.width / 2, bbox.ymin + bbox.height / 2

            x_min = bbox.xmin
            x_max = bbox.xmin + bbox.width
            y_min = bbox.ymin
            y_max = bbox.ymin + bbox.height

            # **Move bounding box center to (0,0) before scaling**
            x_min -= cx
            x_max -= cx
            y_min -= cy
            y_max -= cy

            # **Scale while keeping center fixed**
            x_min *= scale_factor
            x_max *= scale_factor
            y_min *= scale_factor
            y_max *= scale_factor

            # **Move back to original center**
            x_min += cx
            x_max += cx
            y_min += cy
            y_max += cy

            x_min *= W
            x_max *= W
            y_min *= H
            y_max *= H

            # **Convert to absolute pixel values**
            cx, cy = cx * W, cy * H

            # **Ensure integer indices for cropping**
            x_min, x_max = int(round(x_min)), int(round(x_max))
            y_min, y_max = int(round(y_min)), int(round(y_max))

            # **Step 1: Expand the bounding box to be square**
            box_width = x_max - x_min
            box_height = y_max - y_min
            max_side = max(box_width, box_height)  # Make the bounding box square

            # **Re-center the bounding box while making it square**
            x_min = int(round(cx - max_side / 2))
            x_max = int(round(cx + max_side / 2))
            y_min = int(round(cy - max_side / 2))
            y_max = int(round(cy + max_side / 2))

            # **Step 2: Clamp the modified bounding box to image bounds**
            x_min_clamped, x_max_clamped = max(0, x_min), min(W, x_max)
            y_min_clamped, y_max_clamped = max(0, y_min), min(H, y_max)

            # **Extract the portion inside the image bounds**
            crop = image_batch[i][:, y_min_clamped:y_max_clamped, x_min_clamped:x_max_clamped]

            # **Step 3: Compute padding needed for out-of-bounds areas**
            pad_top = abs(y_min - y_min_clamped) if y_min < 0 else 0
            pad_bottom = abs(y_max - y_max_clamped) if y_max > H else 0
            pad_left = abs(x_min - x_min_clamped) if x_min < 0 else 0
            pad_right = abs(x_max - x_max_clamped) if x_max > W else 0

            # **Step 4: Apply padding where necessary**
            crop_padded = F.pad(crop, (pad_left, pad_right, pad_top, pad_bottom), value=padding_value)
            _, _, padded_W = crop_padded.shape
            crop_padded = T.functional.resize(crop_padded, (target_size, target_size), antialias=False)

            # **Step 5: Record scaled padding**
            crop_scale = target_size / padded_W

            # **Record results**
            cropped_images.append(crop_padded.cpu())
            bboxes.append((x_min_clamped, y_min_clamped, x_max_clamped, y_max_clamped))  # Bounding box before clamping
            centers.append((int(cx), int(cy)))  # Face center
            paddings.append((int(crop_scale * pad_top), int(crop_scale * pad_bottom), int(crop_scale * pad_left), int(crop_scale * pad_right)))
        else:
            # No face detected, return a blank image instead of an empty batch
            blank_image = torch.full((3, target_size, target_size), padding_value, dtype=torch.float32)
            cropped_images.append(blank_image)  # Keep batch shape consistent
            bboxes.append(None)
            centers.append(None)
            paddings.append(None)

    if not cropped_images:
        raise ValueError("No valid images found in batch!")

    cropped_images = torch.stack(cropped_images)  # Ensure all images have the same shape

    # If the input was a single image, remove batch dimension
    """if B == 1:
        return cropped_images.squeeze(0), bboxes[0], centers[0], paddings[0]"""

    return cropped_images, bboxes, centers, paddings

def load_dataloaders(config):
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_VOCASET, val_dataset_VOCASET, test_dataset_VOCASET = get_datasets_VOCASET(config)
    dataset_percentages = {
        'VOCASET': 1.0
    }
    
    train_dataset = train_dataset_VOCASET
    sampler = MixedDatasetBatchSampler([
                                        len(train_dataset_VOCASET)
                                        ], 
                                       list(dataset_percentages.values()), 
                                       config.train.batch_size, len(train_dataset_VOCASET))
    def collate_fn(batch):
        combined_batch = {}
        for key in batch[0].keys():
            combined_batch[key] = [b[key] for b in batch]

        return combined_batch
    
    val_dataset = val_dataset_VOCASET
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=sampler, num_workers=config.train.batch_size, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return train_loader, val_dataset, test_dataset_VOCASET

if __name__ == '__main__':
    config = parse_args()

    os.makedirs(config.train.log_path, exist_ok=True)
    vocaset_save_path = os.path.join(config.train.log_path, 'vocaset_all')
    os.makedirs(vocaset_save_path, exist_ok=True)

    from config import parse_args
    cfg = parse_args()
    cfg.exp_name = cfg.output_dir

    train_loader, val_loader, _ = load_dataloaders(config)

    def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Initialize models
    """tater = SmirkEncoder(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    loaded_state_dict = torch.load(config.resume, map_location=config.device)
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if k.startswith('smirk_encoder')}
    filtered_state_dict = strip_exact_prefix(filtered_state_dict, "smirk_encoder.")
    tater.load_state_dict(filtered_state_dict)
    tater.device = config.device
    force_model_to_device(tater, tater.device)
    tater.eval()"""

    spectre = SPECTRE(cfg)
    spectre.device = config.device
    force_model_to_device(spectre, spectre.device)
    spectre.eval()

    """flame = FLAME(n_exp=config.arch.num_expression, n_shape=config.arch.num_shape)
    flame = flame.to(config.device)
    force_model_to_device(flame, config.device)
    flame.eval()"""

    renderer = Renderer(render_full_head=False)
    force_model_to_device(renderer, config.device)
    renderer.eval()

    # Face Detection
    if FACE_CROP:
        face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Main df capturing all sampled items
    columns = ["file", "problematic", "S2M_Dist"]
    main_results_df = pd.DataFrame(columns=columns)

    # Process validation set
    for segment_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        if segment_idx <= 25:
            continue
        """if (segment_idx >= 25):
            break"""
        if len(batch["img"]) == 0:
            continue

        subbatch_start = 0
        fps = batch["fps"][0].item()
        sample_rate = batch["audio_sample_rate"][0].item()

        fine_results_df = pd.DataFrame(columns=["S2M_Dist"])
        with tqdm(total=len(batch["img"]), desc="Processing Batches", unit="batch", dynamic_ncols=True) as pbar:
            while subbatch_start < len(batch["img"]):
                imgs = batch["img"][subbatch_start:subbatch_start + SEG_LEN].to(config.device)

                if FACE_CROP:
                    cropped_imgs, bboxes, _, paddings = detect_face_and_crop(imgs, face_detector)
                    cropped_imgs = cropped_imgs.to(config.device) 
                else:
                    cropped_imgs = imgs
                    bboxes = [None] * len(imgs)
                    paddings = [None] * len(imgs)

                with torch.no_grad():
                    """all_params = tater(cropped_imgs)
                    flame_output = flame.forward(all_params)
                    rendered_imgs = renderer.forward(flame_output['vertices'], all_params['cam'])["rendered_img"]"""

                    all_params = spectre.encode(cropped_imgs)
                    flame_output = spectre.flame.forward(shape_params=all_params[0]['shape'], expression_params=all_params[0]['exp'],
                                                        pose_params=all_params[0]['pose'])
                    rendered_imgs = renderer.forward(flame_output[0], all_params[0]['cam'])["rendered_img"]


                    # **Predictions are Point Clouds**
                    overlap_removed = 0 if subbatch_start == 0 else OVERLAP
                    meshes = flame_output[0][overlap_removed:]  # (B, N, 3)

                    # **Normalize Predictions**
                    meshes -= meshes.mean(dim=1, keepdim=True)  # Center
                    scale = meshes.norm(dim=2, keepdim=True).amax(dim=1, keepdim=True)[0]
                    meshes /= scale

                    # **Convert GT Meshes from batch**
                    gt_meshes = batch["meshes"][subbatch_start:subbatch_start + SEG_LEN]  # Full GT meshes
                    gt_meshes = gt_meshes[overlap_removed:]

                    # **Store results**
                    scan_to_mesh_dists = []

                    # **Process ICP in Smaller Batches**
                    for i in range(0, len(meshes), ICP_BATCH_SIZE):
                        batch_meshes = meshes[i:i + ICP_BATCH_SIZE].to(config.device)  # Move batch to GPU
                        batch_gt_meshes = [gt_meshes[j].to(config.device) for j in range(i, min(i + ICP_BATCH_SIZE, len(gt_meshes)))]  # Move GT batch

                        # Convert Predictions & GT to Pointclouds WITHOUT PADDING
                        source_pcd = Pointclouds(points=[batch_meshes[j] for j in range(len(batch_meshes))])
                        target_pcd = Pointclouds(points=[batch_gt.verts_packed() for batch_gt in batch_gt_meshes])

                        # Run ICP for Alignment (No Masking)
                        _, _, aligned_meshes, _, _ = iterative_closest_point(
                            source_pcd, target_pcd, max_iterations=500
                        )

                        # Compute Scan-to-Mesh Distance in the current batch
                        batch_dists = [
                            point_mesh_face_distance(batch_gt_meshes[j], aligned_meshes[j]).detach().cpu().numpy()
                            for j in range(len(aligned_meshes))
                        ]

                        scan_to_mesh_dists.extend(batch_dists)  # Store results

                        # Free up memory
                        del batch_meshes, batch_gt_meshes, aligned_meshes, source_pcd, target_pcd
                        torch.cuda.empty_cache()

                    # **Append Results to DataFrame**
                    new_data_df = pd.DataFrame({"S2M_Dist": scan_to_mesh_dists.tolist()})
                    fine_results_df = pd.concat([fine_results_df, new_data_df], ignore_index=True)
                    
                    # **Progress Updates**
                    pbar.update(SEG_LEN - OVERLAP)
                    subbatch_start += SEG_LEN - OVERLAP

            avg_dist = fine_results_df["S2M_Dist"].mean()
            dir_name = batch["data_dir"]
            print(f"Chamfer Distance for {dir_name} (Batch-wise, Aligned): {avg_dist}")

            main_results_df = pd.concat([main_results_df, pd.DataFrame([[batch["data_dir"], batch["problematic"], avg_dist]], columns=columns)], ignore_index=True)
            fine_results_df.to_csv(f"{vocaset_save_path}/{Path(dir_name).stem}.csv")
        
        main_results_df.to_csv(f"{vocaset_save_path}/main_results_0_9.csv")

