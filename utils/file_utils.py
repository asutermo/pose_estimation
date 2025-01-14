import base64
import logging
from urllib.parse import urlparse
import magic
import os

from PIL import Image
import httpx

from utils.log_utils import config_logs

__all__ = ["get_mime_type", "encode_image", "is_url", "get_mime_type"]

config_logs()
logger = logging.getLogger(__name__)

def get_mime_type(path: str) -> str:
    """Get the mime type of the file"""
    if is_url(path):
        return httpx.head(path).headers.get("Content-Type")
    else:
        return magic.from_file(path, mime=True)

def is_url(path: str) -> bool:
    """Check if the path is a url"""
    return urlparse(path).netloc != ""

def get_pil_image(path: str, is_url: bool) -> Image.Image:
    """Get the PIL image from the file path"""
    if is_url:
        return  Image.open(httpx.get(path))
    
    return Image.open(path)
    

def encode_image(image_path: str) -> str:
    """Encode the image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
