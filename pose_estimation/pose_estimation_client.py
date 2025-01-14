import os
import httpx
import torch
import numpy as np
import supervision as sv

from PIL import Image

__all__ = ['PoseEstimationClient']

from transformers import(
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation
)

class PoseEstimationClient():
    def __init__(self, autoprocessor_model: str = "PekingU/rtdetr_r50vd_coco_o365", rtdetr_model: str = "PekingU/rtdetr_r50vd_coco_o365",  image_processor_model: str = "usyd-community/vitpose-base-simple", pose_estimation_model: str = "usyd-community/vitpose-base-simple"): 

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_image_processor = AutoProcessor.from_pretrained(autoprocessor_model)
        self.person_model = RTDetrForObjectDetection.from_pretrained(rtdetr_model, device_map=self.device)
        self.image_processor = AutoProcessor.from_pretrained(image_processor_model)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(pose_estimation_model, device_map=self.device)

    def process_image(self, image_path_or_url: str):
        if not image_path_or_url:
            image_path_or_url = "http://images.cocodataset.org/val2017/000000000139.jpg"

        if os.path.exists(image_path_or_url):
            image = Image.open(image_path_or_url)
        else:
            image = Image.open(httpx.get(image_path_or_url))

        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.person_model(**inputs)

        # Post process. This is only single image at the moment
        results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
        )[0]
     
        # Extract the bounding box 
        person_boxes_xyxy = results["boxes"][results["labels"] == 0]
        person_boxes_xyxy = person_boxes_xyxy.cpu().numpy()

        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        person_boxes_xyxy[:, 2] = person_boxes_xyxy[:, 2] - person_boxes_xyxy[:, 0]
        person_boxes_xyxy[:, 3] = person_boxes_xyxy[:, 3] - person_boxes_xyxy[:, 1]
        inputs = self.image_processor(image, boxes=[person_boxes_xyxy], return_tensors="pt").to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        
        pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes_xyxy])
        image_pose_result = pose_results[0]  # results for first image

        xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
        scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

        key_points = sv.KeyPoints(
            xy=xy, confidence=scores
        )

        edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.GREEN,
            thickness=1
        )
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.RED,
            radius=2
        )
        annotated_frame = edge_annotator.annotate(
            scene=image.copy(),
            key_points=key_points
        )
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=key_points
        )
