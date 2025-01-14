import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

# from pydantic import BaseModel  # type: ignore
from pose_estimation.pose_estimation_client import PoseEstimationClient
from utils.image_utils import encode_image, is_image
from utils.log_utils import config_logs

config_logs()
logger = logging.getLogger(__name__)

client = PoseEstimationClient()
client.process_image("")