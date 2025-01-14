import logging

# from pydantic import BaseModel  # type: ignore
from pose_estimation.pose_estimation_client import PoseEstimationClient
from utils.log_utils import config_logs

config_logs()
logger = logging.getLogger(__name__)

client = PoseEstimationClient()
client.process_image("")
