import os
from typing import List, Tuple, Union
from PIL import Image
import imageio.v2 as imageio
import numpy as np


def get_filenames(input_dir: str) -> List[str]:
    """
    Get the list of image filenames from the input directory.
    Args:
        input_dir (str): Directory containing the images.
    Returns:
        List[str]: List of image filenames.
    """
    return [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".jpg")]


def load_image(
        path: str,
        crop: Tuple[int, int, int, int] = None,
        convert: str = "RGB") -> Union[np.ndarray, None]:
    """
    Load an image from the given path and convert it to a numpy array.
    Args:
        path (str): Path to the image.
        crop (Tuple[int, int, int, int]): Coordinates for cropping (left, upper, right, lower).
        convert (str): Color mode to convert the image to (default is "RGB").
    Returns:
        np.ndarray: Numpy array of the image.
    """
    try:
        with Image.open(path) as img:
            img = img.convert(convert)
            if crop:
                img = img.crop(crop)
            return np.array(img)
    except Exception as e:
        print(f"[!] Failed to load image {path}: {e}")
        return None


def patchify(
        img: np.ndarray,
        patch_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """
    Patchify the image into patches of size patch_sizes.
    Args:
        img (np.ndarray): The image to patchify.
        patch_sizes (Tuple[int, int]): The size of the patches.
    Returns:
        List[np.ndarray]: A list of patches.
    """
    h, w = patch_size
    patches = []
    for i in range(0, img.shape[0], h):
        for j in range(0, img.shape[1], w):
            patch = img[i:i+h, j:j+w]
            if patch.shape[0] == h and patch.shape[1] == w:
                patches.append(patch)
    return patches


def save_patches(
        patches: List[np.ndarray],
        output_dir: str,
        starting_index: int) -> None:
    """Save the synthetic clouds to the output directory.
    Args:
        patches: The list of patches.
        output_dir: The output directory.
        starting_index: The starting index for naming the files.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, patch in enumerate(patches):
        filename = os.path.join(output_dir, f"patch_{starting_index + i:05d}.jpg")
        imageio.imwrite(filename, patch)