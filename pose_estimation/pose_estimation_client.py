import json
import logging
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import supervision as sv  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from PIL import Image
from transformers import (  # type: ignore
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

from utils.log_utils import config_logs

__all__ = ["PoseEstimationClient"]

config_logs()
logger = logging.getLogger(__name__)


class PoseEstimationClient:
    def __init__(
        self,
        autoprocessor_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        rtdetr_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        image_processor_model: str = "usyd-community/vitpose-base-simple",
        pose_estimation_model: str = "usyd-community/vitpose-base-simple",
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.person_image_processor = AutoProcessor.from_pretrained(autoprocessor_model)
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            rtdetr_model, device_map=self.device
        )
        self.image_processor = AutoProcessor.from_pretrained(image_processor_model)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            pose_estimation_model, device_map=self.device
        )

    @torch.inference_mode()
    def process_image(
        self, image: Image.Image, output_file: Optional[str] = None
    ) -> tuple[Image.Image, list[dict]]:
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        results = self.person_image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=0.3,
        )
        result = results[0]

        # Extract the bounding box
        person_boxes_xyxy = result["boxes"][result["labels"] == 0]
        person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()

        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        person_boxes = person_boxes_xyxy.copy()
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

        inputs = self.image_processor(
            image, boxes=[person_boxes], return_tensors="pt"
        ).to(self.device)

        # MOE
        if self.pose_model.config.backbone_config.num_experts > 1:
            dataset_index = torch.tensor([0] * len(inputs["pixel_values"]))
            dataset_index = dataset_index.to(inputs["pixel_values"].device)
            inputs["dataset_index"] = dataset_index

        # forward pass
        with torch.no_grad():
            outputs = self.pose_model(**inputs)

        pose_results = self.image_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )
        image_pose_result = pose_results[0]  # results for first image

        human_readable_results = []
        for i, person_pose in enumerate(image_pose_result):
            data = {
                "person_id": i,
                "bbox": person_pose["bbox"].numpy().tolist(),
                "keypoints": [],
            }
            for keypoint, label, score in zip(
                person_pose["keypoints"],
                person_pose["labels"],
                person_pose["scores"],
                strict=True,
            ):
                keypoint_name = self.pose_model.config.id2label[label.item()]
                x, y = keypoint
                data["keypoints"].append(
                    {
                        "name": keypoint_name,
                        "x": x.item(),
                        "y": y.item(),
                        "score": score.item(),
                    }
                )
            human_readable_results.append(data)

        xy = (
            torch.stack([pose_result["keypoints"] for pose_result in image_pose_result])
            .cpu()
            .numpy()
        )
        scores = (
            torch.stack([pose_result["scores"] for pose_result in image_pose_result])
            .cpu()
            .numpy()
        )

        keypoints = sv.KeyPoints(xy=xy, confidence=scores)
        detections = sv.Detections(xyxy=person_boxes_xyxy)

        edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=1)
        vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=2)
        bounding_box_annotator = sv.BoxAnnotator(
            color=sv.Color.WHITE, color_lookup=sv.ColorLookup.INDEX, thickness=1
        )

        annotated_frame = image.copy()

        # annotate boundg boxes
        annotated_frame = bounding_box_annotator.annotate(
            scene=image.copy(), detections=detections
        )

        # annotate edges and verticies
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame, key_points=keypoints
        )
        annotated_result = vertex_annotator.annotate(
            scene=annotated_frame, key_points=keypoints
        )

        if output_file:
            annotated_result.save(output_file)
            output_json = os.path.join(
                os.path.dirname(output_file), f"{os.path.basename(output_file)}.json"
            )
            js = json.dumps({"path": output_file, "results": human_readable_results})
            with open(output_json, "w") as f:
                f.write(js)
            logger.info(f"{output_file}: {js}")

        return (
            annotated_result,
            human_readable_results,
        )

    def process_video(
        self,
        video_path: str,
        max_num_frames: int = 60,
        output_file: Optional[str] = None,
    ) -> str:
        cap = cv2.VideoCapture(video_path)  # type: ignore
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        o_file = (
            open(output_file, "w")
            if output_file
            else tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        )
        with o_file as out_file:
            writer = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))  # type: ignore
            for _ in tqdm.auto.tqdm(range(min(max_num_frames, num_frames))):
                ok, frame = cap.read()
                if not ok:
                    logger.warn("Bad frame found")
                    break
                rgb_frame = frame[:, :, ::-1]
                annotated_frame, _ = self.process_image(Image.fromarray(rgb_frame))
                writer.write(np.asarray(annotated_frame)[:, :, ::-1])
            writer.release()
        cap.release()
        return out_file.name
