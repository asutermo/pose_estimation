import argparse
import logging
import os

# from pydantic import BaseModel  # type: ignore
from pose_estimation.pose_estimation_client import PoseEstimationClient
from utils import file_utils
from utils.log_utils import config_logs

config_logs()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation Client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--paths", nargs="+", required=True, help="Images (Url or Path), Directories or Video Paths to process")    
    parser.add_argument("-o", "--output", type=str, default="output", help="Output directory to save the processed files")

    args = parser.parse_args()
    client = PoseEstimationClient()

    for path in args.paths:
        logger.info("Processing path: %s", path)
        if file_utils.is_url(path):
            mime_type = file_utils.get_mime_type(path)
            if mime_type.startswith("image/"):
                image = file_utils.get_pil_image(path, True)
                res = client.process_image(image)
                print(res)
            elif mime_type.startswith("video/"):
                print(client.process_video(path, 60))
            else:
                logger.error("Unsupported file type: %s", mime_type)
        elif os.path.exists(path):
            # file or dir
            if os.path.isfile(path):
                file_mime = file_utils.get_mime_type(path)
                if file_mime.startswith("image/"):
                    image = file_utils.get_pil_image(path, False)
                    res = client.process_image(image)
                    print(res)
                elif file_mime.startswith("video/"):
                    print(client.process_video(path, 60))
                else:
                    logger.info(f"Unsupported file type: %s", )
              
            else:
                pass