"""Compare RGB-only metric depth models against URFD Kinect reference depth.

This is an offline experiment. Kinect depth is used only as evaluation ground
truth; the model inputs are RGB frames. The current sample compares cam0 fall-01
and adl-01 at regular intervals. Its outputs are kept separate from the KNN
classifier experiment.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = ROOT / ".depth_runtime"
sys.path.insert(0, str(RUNTIME_PACKAGES))

import torch
import torch.nn.functional as functional


DA_REPO = ROOT / "third_party" / "Depth-Anything-V2"
METRIC3D_REPO = ROOT / "third_party" / "Metric3D"
sys.path.insert(0, str(DA_REPO / "metric_depth"))
sys.path.insert(0, str(METRIC3D_REPO))

from depth_anything_v2.dpt import DepthAnythingV2
import hubconf as metric3d_hub


RGB_ROOT = ROOT / "data" / "urfd" / "raw" / "UR_fall_detection_dataset_cam0_rgb"
DEPTH_ROOT = ROOT / "data" / "urfd" / "depth_sample"
POSE_ROOT = ROOT / "outputs" / "poses"
CHECKPOINT_ROOT = ROOT / "models" / "depth_models"
OUTPUT_ROOT = ROOT / "outputs" / "depth_model_comparison"

FRAME_STRIDE = 15
DEPTH_PATCH_RADIUS = 2
VISIBILITY_MINIMUM = 0.5
DEPTH_MAX_METRES = 7.0

# Metric3D's canonical-space conversion requires the RGB camera focal length.
# URFD does not provide it in the downloaded metadata. This provisional value
# is only for the sample comparison; calibrate the deployment camera later.
METRIC3D_FOCAL_PX = 525.0

DEPTH_SCALE_MM = {
    "fall-": {"cam0": 6000.0},
    "adl-": {"cam0": 7000.0},
}


def official_depth_scale(sequence_id, camera_id="cam0"):
    for prefix, camera_scales in DEPTH_SCALE_MM.items():
        if sequence_id.startswith(prefix) and camera_id in camera_scales:
            return camera_scales[camera_id]
    raise ValueError(f"No URFD depth scale for {sequence_id}/{camera_id}.")


def load_pose_rows(sequence_id):
    path = POSE_ROOT / f"{sequence_id}-pose-landmarks.csv"
    if not path.exists():
        raise FileNotFoundError(f"MediaPipe pose CSV not found:\n{path}")
    table = pd.read_csv(path)
    if "time_ms" not in table:
        table["time_ms"] = np.nan
    return table


def sampled_frames(rgb_folder):
    paths = sorted(rgb_folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No RGB frames found in:\n{rgb_folder}")
    selected = list(range(0, len(paths), FRAME_STRIDE))
    if selected[-1] != len(paths) - 1:
        selected.append(len(paths) - 1)
    return [(index + 1, paths[index]) for index in selected]


def create_depth_anything_model(checkpoint_path):
    config = {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "max_depth": 20,
    }
    model = DepthAnythingV2(**config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    return model.eval()


def create_metric3d_model(checkpoint_path):
    model = metric3d_hub.metric3d_vit_small(pretrain=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.eval()


def metric3d_depth_map(model, bgr_image):
    """Run Metric3D small and convert canonical depth using the provisional fx."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    input_height, input_width = 616, 1064
    resize_scale = min(input_height / height, input_width / width)
    resized = cv2.resize(
        rgb,
        (int(width * resize_scale), int(height * resize_scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    focal_scaled = METRIC3D_FOCAL_PX * resize_scale

    resized_height, resized_width = resized.shape[:2]
    pad_height = input_height - resized_height
    pad_width = input_width - resized_width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    resized = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(123.675, 116.28, 103.53),
    )

    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0)

    with torch.inference_mode():
        predicted, _, _ = model.inference({"input": tensor})

    predicted = predicted.squeeze()
    predicted = predicted[top : predicted.shape[0] - bottom, left : predicted.shape[1] - right]
    predicted = functional.interpolate(
        predicted[None, None, :, :],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    predicted = predicted.numpy() * (focal_scaled / 1000.0)
    return np.clip(predicted, 0.0, DEPTH_MAX_METRES)


def depth_metrics(predicted, ground_truth, mask):
    p = predicted[mask].astype(np.float64)
    g = ground_truth[mask].astype(np.float64)
    if not len(g):
        return {"abs_rel": np.nan, "rmse_m": np.nan, "delta1": np.nan, "samples": 0}
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "samples": int(len(g)),
    }


def sample_joint_depths(pose_rows, frame_number, predicted, ground_truth, visibility_minimum):
    rows = pose_rows[pose_rows["frame_number"] == frame_number]
    height, width = predicted.shape
    result = {}
    for row in rows.itertuples(index=False):
        visibility = getattr(row, "visibility", np.nan)
        if not np.isfinite(visibility) or visibility < visibility_minimum:
            continue
        x = int(np.clip(round(row.x_normalized * (width - 1)), 0, width - 1))
        y = int(np.clip(round(row.y_normalized * (height - 1)), 0, height - 1))
        x0, x1 = max(0, x - DEPTH_PATCH_RADIUS), min(width, x + DEPTH_PATCH_RADIUS + 1)
        y0, y1 = max(0, y - DEPTH_PATCH_RADIUS), min(height, y + DEPTH_PATCH_RADIUS + 1)
        gt_patch = ground_truth[y0:y1, x0:x1]
        gt_values = gt_patch[(gt_patch > 0.0) & (gt_patch <= DEPTH_MAX_METRES)]
        p_patch = predicted[y0:y1, x0:x1]
        p_values = p_patch[np.isfinite(p_patch) & (p_patch > 0.0)]
        if gt_values.size == 0 or p_values.size == 0:
            continue
        result[row.landmark_name] = {
            "pred_m": float(np.median(p_values)),
            "gt_m": float(np.median(gt_values)),
        }
    return result


def main():
    torch.set_num_threads(2)
    da_checkpoint = CHECKPOINT_ROOT / "depth_anything_v2_metric_hypersim_vits.pth"
    metric3d_checkpoint = CHECKPOINT_ROOT / "metric_depth_vit_small_800k.pth"
    for checkpoint in (da_checkpoint, metric3d_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Depth model checkpoint not found:\n{checkpoint}")

    # Evaluate only sequences for which all three required inputs are present:
    # RGB frames, Kinect depth reference frames, and MediaPipe pose landmarks.
    available_sequences = []
    for depth_folder in sorted(DEPTH_ROOT.glob("*-cam0-d")):
        sequence_id = depth_folder.name.removesuffix("-cam0-d")
        has_depth_images = next(depth_folder.rglob("*-cam0-d-*.png"), None) is not None
        if (
            (RGB_ROOT / f"{sequence_id}-cam0-rgb").is_dir()
            and (POSE_ROOT / f"{sequence_id}-pose-landmarks.csv").is_file()
            and has_depth_images
        ):
            available_sequences.append(sequence_id)
    if not available_sequences:
        raise FileNotFoundError(
            "No sequences have matching RGB, Kinect depth, and pose files."
        )

    print(f"Sequences with all required inputs: {', '.join(available_sequences)}", flush=True)
    print(f"Sampling every {FRAME_STRIDE} frames, plus each sequence's last frame.", flush=True)
    print("Loading Depth Anything V2 Metric Indoor Small...", flush=True)
    da_model = create_depth_anything_model(da_checkpoint)
    print("Loading Metric3D v2 ViT-Small...", flush=True)
    metric3d_model = create_metric3d_model(metric3d_checkpoint)
    models = {
        "DepthAnythingV2_MetricIndoor_Small": lambda image: da_model.infer_image(image),
        "Metric3D_v2_ViT_Small": lambda image: metric3d_depth_map(metric3d_model, image),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame_records = []
    joint_records = []
    velocity_records = []

    for sequence_id in available_sequences:
        rgb_folder = RGB_ROOT / f"{sequence_id}-cam0-rgb"
        depth_folder = DEPTH_ROOT / f"{sequence_id}-cam0-d"
        pose_rows = load_pose_rows(sequence_id)
        scale_mm = official_depth_scale(sequence_id)
        previous_joint_depths = {name: {} for name in models}

        for frame_number, rgb_path in sampled_frames(rgb_folder):
            expected_depth_name = f"{sequence_id}-cam0-d-{frame_number:03d}.png"
            depth_path = next(depth_folder.rglob(expected_depth_name), None)
            if depth_path is None:
                raise FileNotFoundError(
                    f"Matching Kinect depth frame not found for {sequence_id} "
                    f"frame {frame_number}: {expected_depth_name}"
                )
            bgr_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if bgr_image is None or raw_depth is None:
                raise RuntimeError(f"Could not read matched RGB/depth frame {sequence_id} {frame_number}.")
            if bgr_image.shape[:2] != raw_depth.shape[:2]:
                raise RuntimeError(f"RGB/depth dimensions differ for {sequence_id} frame {frame_number}.")

            ground_truth = raw_depth.astype(np.float32) * (scale_mm / 65535.0) / 1000.0
            ground_truth_mask = (raw_depth > 0) & (ground_truth <= DEPTH_MAX_METRES)
            pose_frame = pose_rows[pose_rows["frame_number"] == frame_number]
            time_ms = float(pose_frame["time_ms"].dropna().iloc[0]) if pose_frame["time_ms"].notna().any() else np.nan

            for model_name, infer in models.items():
                predicted = np.asarray(infer(bgr_image), dtype=np.float32)
                if predicted.shape != ground_truth.shape:
                    predicted = cv2.resize(predicted, (ground_truth.shape[1], ground_truth.shape[0]))
                predicted = np.clip(predicted, 0.05, DEPTH_MAX_METRES)
                metrics = depth_metrics(predicted, ground_truth, ground_truth_mask)
                joint_values = sample_joint_depths(
                    pose_rows, frame_number, predicted, ground_truth, VISIBILITY_MINIMUM
                )
                joint_pred = np.asarray([item["pred_m"] for item in joint_values.values()], dtype=float)
                joint_gt = np.asarray([item["gt_m"] for item in joint_values.values()], dtype=float)
                joint_mask = np.isfinite(joint_pred) & np.isfinite(joint_gt) & (joint_gt > 0)
                joint_result = depth_metrics(joint_pred, joint_gt, joint_mask) if joint_mask.any() else {
                    "abs_rel": np.nan, "rmse_m": np.nan, "delta1": np.nan, "samples": 0
                }

                frame_records.append({
                    "sequence": sequence_id,
                    "frame_number": frame_number,
                    "time_ms": time_ms,
                    "model": model_name,
                    "full_abs_rel": metrics["abs_rel"],
                    "full_rmse_m": metrics["rmse_m"],
                    "full_delta1": metrics["delta1"],
                    "valid_depth_pixels": metrics["samples"],
                    "joint_abs_rel": joint_result["abs_rel"],
                    "joint_rmse_m": joint_result["rmse_m"],
                    "joint_delta1": joint_result["delta1"],
                    "visible_joints_scored": joint_result["samples"],
                })

                for landmark_name, values in joint_values.items():
                    joint_records.append({
                        "sequence": sequence_id,
                        "frame_number": frame_number,
                        "time_ms": time_ms,
                        "model": model_name,
                        "landmark_name": landmark_name,
                        "predicted_depth_m": values["pred_m"],
                        "kinect_reference_depth_m": values["gt_m"],
                        "absolute_error_m": abs(values["pred_m"] - values["gt_m"]),
                    })
                    previous = previous_joint_depths[model_name].get(landmark_name)
                    if previous and np.isfinite(time_ms) and time_ms > previous["time_ms"]:
                        elapsed_s = (time_ms - previous["time_ms"]) / 1000.0
                        pred_velocity = (values["pred_m"] - previous["pred_m"]) / elapsed_s
                        gt_velocity = (values["gt_m"] - previous["gt_m"]) / elapsed_s
                        velocity_records.append({
                            "sequence": sequence_id,
                            "frame_number": frame_number,
                            "model": model_name,
                            "landmark_name": landmark_name,
                            "time_delta_s": elapsed_s,
                            "predicted_depth_velocity_mps": pred_velocity,
                            "kinect_depth_velocity_mps": gt_velocity,
                            "absolute_velocity_error_mps": abs(pred_velocity - gt_velocity),
                        })
                    previous_joint_depths[model_name][landmark_name] = {
                        "time_ms": time_ms,
                        "pred_m": values["pred_m"],
                        "gt_m": values["gt_m"],
                    }

            print(f"Compared {sequence_id} frame {frame_number}/{len(list(rgb_folder.glob('*.png')))}", flush=True)

    frame_table = pd.DataFrame(frame_records)
    joint_table = pd.DataFrame(joint_records)
    velocity_table = pd.DataFrame(velocity_records)
    summary = frame_table.groupby("model", as_index=False).agg(
        frames_scored=("frame_number", "count"),
        full_abs_rel_mean=("full_abs_rel", "mean"),
        full_rmse_m_mean=("full_rmse_m", "mean"),
        full_delta1_mean=("full_delta1", "mean"),
        joint_abs_rel_mean=("joint_abs_rel", "mean"),
        joint_rmse_m_mean=("joint_rmse_m", "mean"),
        joint_delta1_mean=("joint_delta1", "mean"),
        visible_joints_scored=("visible_joints_scored", "sum"),
    )
    velocity_summary = velocity_table.groupby("model", as_index=False).agg(
        joint_velocity_pairs=("absolute_velocity_error_mps", "count"),
        mean_joint_depth_velocity_error_mps=("absolute_velocity_error_mps", "mean"),
        median_joint_depth_velocity_error_mps=("absolute_velocity_error_mps", "median"),
    )
    summary = summary.merge(velocity_summary, on="model", how="left")

    sequence_summary = frame_table.groupby(["sequence", "model"], as_index=False).agg(
        frames_scored=("frame_number", "count"),
        full_abs_rel_mean=("full_abs_rel", "mean"),
        full_rmse_m_mean=("full_rmse_m", "mean"),
        full_delta1_mean=("full_delta1", "mean"),
        joint_abs_rel_mean=("joint_abs_rel", "mean"),
        joint_rmse_m_mean=("joint_rmse_m", "mean"),
        joint_delta1_mean=("joint_delta1", "mean"),
        visible_joints_scored=("visible_joints_scored", "sum"),
    )
    sequence_velocity_summary = velocity_table.groupby(
        ["sequence", "model"], as_index=False
    ).agg(
        joint_velocity_pairs=("absolute_velocity_error_mps", "count"),
        mean_joint_depth_velocity_error_mps=("absolute_velocity_error_mps", "mean"),
        median_joint_depth_velocity_error_mps=("absolute_velocity_error_mps", "median"),
    )
    sequence_summary = sequence_summary.merge(
        sequence_velocity_summary, on=["sequence", "model"], how="left"
    )

    frame_table.to_csv(OUTPUT_ROOT / "frame_depth_metrics.csv", index=False)
    joint_table.to_csv(OUTPUT_ROOT / "joint_depth_metrics.csv", index=False)
    velocity_table.to_csv(OUTPUT_ROOT / "joint_depth_velocity_metrics.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "depth_model_summary.csv", index=False)
    sequence_summary.to_csv(
        OUTPUT_ROOT / "depth_model_summary_by_sequence.csv", index=False
    )

    print("\nDepth-model sample comparison summary:")
    print(summary.to_string(index=False))
    print("\nPer-sequence summary:")
    print(sequence_summary.to_string(index=False))
    print(f"Saved separate depth-model results under:\n{OUTPUT_ROOT}")
    print(
        "Metric3D uses the provisional 525 px RGB focal length in this sample. "
        "Calibrate the actual deployment camera before treating its depths as metric."
    )


if __name__ == "__main__":
    main()
