import argparse
import logging
import os

# from pydantic import BaseModel  # type: ignore
from pose_estimation.pose_estimation_client import PoseEstimationClient
from utils import file_utils
from utils.log_utils import config_logs

config_logs()
logger = logging.getLogger(__name__)


def process_based_on_mime_type(path: str, output_path: str, max_frames: int) -> None:
    mime_type = file_utils.get_mime_type(path)

    if mime_type.startswith("image/"):
        image = file_utils.get_pil_image(path, file_utils.is_url(path))
        img_path = os.path.join(output_path, f"annotated_{os.path.basename(path)}")

        _ = client.process_image(image, img_path)
    elif mime_type.startswith("video/"):
        vid_path = os.path.join(output_path, f"annotated_{os.path.basename(path)}")
        _ = client.process_video(path, max_frames, vid_path)
    else:
        logger.error(f"Unsupported file type: {mime_type}")
        return


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
    parser.add_argument(
        "-n",
        "--max_frames",
        type=int,
        default=60,
        help="Max frames to process per video",
    )

    args = parser.parse_args()
    client = PoseEstimationClient()

    os.makedirs(args.output, exist_ok=True)
    for path in args.paths:
        logger.info("Processing path: %s", path)

        if os.path.isdir(path):
            for file in os.listdir(path):
                process_based_on_mime_type(
                    os.path.join(path, file), args.output, args.max_frames
                )
        else:
            process_based_on_mime_type(path, args.output, args.max_frames)
