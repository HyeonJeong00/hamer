from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--video_path', type=str, default='video.mp4', help='Path to input video file')
    parser.add_argument('--out_video_path', type=str, default='out_demo.mp4', help='Output path to save rendered video')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    # parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    start_t = time.time()
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    end_t = time.time()
    print(f"detector loaded, time: {end_t-start_t}")
    # Keypoint detector
    cpm = ViTPoseModel(device)
    print("VIT loaded")

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Open video capture
    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15  # Set fps to 15 for better compatibility

    # Define codec and create VideoWriter object to save output
    out = cv2.VideoWriter(args.out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print("starting vcap")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect humans in frame
        det_out = detector(frame)
        img = frame[:, :, ::-1]  # Convert to RGB

        print("break1")
        start_t = time.time()
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        print("starting vit estimation")

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        end_t = time.time()
        bboxes = []
        is_right = []
        print(f"starting vitpose, time: {end_t-start_t}")
        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            start_t = time.time()
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            print("break2")

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
            end_t = time.time()
        print(f"break3, time: {end_t-start_t}")
        print(f"Number of bboxes: {len(bboxes)}")
        print(f"Left hand keypoints confidence: {sum(left_hand_keyp[:, 2] > 0.5)}")
        print(f"Right hand keypoints confidence: {sum(right_hand_keyp[:, 2] > 0.5)}")

        if len(bboxes) == 0:
            out.write(frame)
            continue
        
        print("break4")

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        print("starting dataset")
        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Lists to accumulate vertices, cam translations, and hands orientation for full frame rendering
        all_verts = []
        all_cam_t = []
        all_right = []

        for batch in dataloader:
            start_t = time.time()
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out_pred = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out_pred['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Add all verts and cams to list
            for n in range(batch['img'].shape[0]):
                verts = out_pred['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

            # Render front view
            if args.full_frame and len(all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

                # Overlay image
                input_fr = frame.astype(np.float32)[:,:,::-1]/255.0
                input_fr = np.concatenate([input_fr, np.ones_like(input_fr[:,:,:1])], axis=2) # Add alpha channel
                input_fr_overlay = input_fr[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                # Convert to uint8 before writing to the video
                input_fr_overlay_uint8 = (255 * input_fr_overlay[:, :, ::-1]).astype(np.uint8)
                out.write(input_fr_overlay_uint8)
            else:
                # If full_frame is False or no vertices are found, write the unmodified frame
                out.write(frame)
            end_t = time.time()
            print(f"single round of Hamer, time: {end_t-start_t}")

    cap.release()
    out.release()


if __name__ == '__main__':
    main()
