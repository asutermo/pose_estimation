import argparse
import logging
import os

# from pydantic import BaseModel  # type: ignore
from pose_estimation.pose_estimation_client import PoseEstimationClient
from utils import file_utils
from utils.log_utils import config_logs

config_logs()
logger = logging.getLogger(__name__)


def process_based_on_mime_type(path: str, output_path: str):
    mime_type = file_utils.get_mime_type(path)
    if mime_type.startswith("image/"):
        image = file_utils.get_pil_image(path, file_utils.is_url(path))
        res = client.process_image(image)
        logger.info(f"{path}: {res}")
    elif mime_type.startswith("video/"):
        res = client.process_video(path, 60)
        logger.info(f"{path} {res}")
    else:
        logger.error(f"Unsupported file type: {mime_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pose Estimation Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        required=True,
        help="Images (Url or Path), Directories or Video Paths to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory to save the processed files",
    )

    args = parser.parse_args()
    client = PoseEstimationClient()

    os.makedirs(args.output, exist_ok=True)
    for path in args.paths:
        logger.info("Processing path: %s", path)

        if os.path.isdir(path):
            for file in os.listdir(path):
                process_based_on_mime_type(os.path.join(path, file), args.output)
        else:
            process_based_on_mime_type(path, args.output)
