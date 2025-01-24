import numpy as np
from PIL import Image


def np_to_pil(image: np.ndarray) -> Image:
    """
    Converts an image in numpy ndarray format with all values normalized between 0 and 1 to a PIL image, integer values between 0 and 255.
    """
    if image.dtype != np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.uint8(np.clip(image * 255, 0, 255))
    return Image.fromarray(image)
