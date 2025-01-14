import base64
import imghdr

__all__ = ["is_image", "encode_image"]


def is_image(file_path: str) -> bool:
    """Determine file type based on the image"""
    return imghdr.what(file_path) is not None


def encode_image(image_path: str) -> str:
    """Encode the image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
