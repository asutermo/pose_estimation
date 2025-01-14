import os
import tempfile

import cv2
import httpx
import numpy as np
import supervision as sv
import torch
import tqdm  # type: ignore
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

__all__ = ["PoseEstimationClient"]


class PoseEstimationClient:
    def __init__(
        self,
        autoprocessor_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        rtdetr_model: str = "PekingU/rtdetr_r50vd_coco_o365",
        image_processor_model: str = "usyd-community/vitpose-base-simple",
        pose_estimation_model: str = "usyd-community/vitpose-base-simple",
    ):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        self, image_or_image_path: str | Image.Image
    ) -> tuple[Image.Image, list[dict]]:
        if isinstance(image, str):
            if os.path.exists(image_or_image_path):
                image = Image.open(image_or_image_path)
            else:
                image = Image.open(httpx.get(image_or_image_path))
        elif isinstance(image, Image.Image):
            image = image_or_image_path
        else:
            raise ValueError(
                f"Invalid input type: {type(image)}. Image must be a path/url or PIL Image"
            )

        inputs = self.person_image_processor(images=image, return_tensors="pt").to(
            self.device
        )

        # forward pass
        with torch.no_grad():
            outputs = self.person_model(**inputs)

        # Post process. This is only single image at the moment
        results = self.person_image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]),
            threshold=0.3,
        )[0]

        # Extract the bounding box
        person_boxes_xyxy = results["boxes"][results["labels"] == 0]
        person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()

        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        person_boxes_xyxy[:, 2] = person_boxes_xyxy[:, 2] - person_boxes_xyxy[:, 0]
        person_boxes_xyxy[:, 3] = person_boxes_xyxy[:, 3] - person_boxes_xyxy[:, 1]
        inputs = self.image_processor(
            image, boxes=[person_boxes_xyxy], return_tensors="pt"
        ).to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.pose_model(**inputs)

        pose_results = self.image_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes_xyxy]
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
        return (
            vertex_annotator.annotate(scene=annotated_frame, key_points=keypoints),
            human_readable_results,
        )


def process_video(self, video_path: str, max_num_frames: int = 60) -> str:
    cap = cv2.VideoCapture(video_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_file:
        writer = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))
        for _ in tqdm.auto.tqdm(range(min(max_num_frames, num_frames))):
            ok, frame = cap.read()
            if not ok:
                break
            rgb_frame = frame[:, :, ::-1]
            annotated_frame, _ = self.process_image(Image.fromarray(rgb_frame))
            writer.write(np.asarray(annotated_frame)[:, :, ::-1])
        writer.release()
    cap.release()
    return out_file.name
