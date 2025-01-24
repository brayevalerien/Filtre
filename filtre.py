import argparse
import os

import clip
import cv2
import numpy as np
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
from yaspin import yaspin
from yaspin.spinners import Spinners

from utils import np_to_pil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(
        prog="Filtre", description="An image dataset filtering script"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the directory containing the images",
        required=True,
    )
    parser.add_argument(
        "--min_faces",
        type=int,
        help="Minimum number of faces that have to be in the images. This makes it possible to remove images without any faces by setting to 1.",
        default=0,
    )
    parser.add_argument(
        "--max_faces",
        type=int,
        help="Maximum number of faces that have to be in the images. This is useful to remove group photos for instance. Set to None to keep the number of faces unbounded.",
        default=None,
    )
    parser.add_argument(
        "--keep_similar",
        type=bool,
        help="If enabled, the similarity filter (that removes images that look the same) is disabled.",
        default=False,
    )
    parser.add_argument(
        "--ignore_aesthetic",
        type=bool,
        help="If enabled, the images won't be sorted by decreasing aesthetic score. This skips the aesthetic score prediction, which greatly reduces runtime, so use that if you don't need aesthetic storting.",
        default=False,
    )
    return parser.parse_args()


def load_images(path: str) -> list[np.ndarray]:
    """
    Loads all the images for a directory, including it's subdirectories.

    Args:
        path (str): path to the root directory.

    Returns:
        list[np.ndarray]: the list of the images found in the root directory and in all its subdirectories.
    """
    valid_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    result = []
    for root, _, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_formats):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is not None:
                    result.append(image / 255.0)
    return result


def filter_by_resolution(
    images: list[np.ndarray], min_resolutions: list[tuple[int, int]]
) -> list[np.ndarray]:
    """
    Removes all images that are under specified resolutions.

    Args:
        images (list[np.ndarray]): list of the images that needs to be filtered.
        min_resolutions (list[tuple[int, int]]): a list of minimum resolutions (in (width, height) format). If an image is smaller than all these resolutions, it will be filtered out.

    Returns:
        list[np.ndarray]: list of the filtered images.
    """
    result = []
    for image in images:
        height, width, _ = image.shape
        if any(
            (min_height < height and min_width < width)
            for min_width, min_height in min_resolutions
        ):
            result.append(image)
    return result


def filter_by_faces(
    images: list[np.ndarray], min_faces: int = 1, max_faces: int = None
) -> list[np.ndarray]:
    """
    Filters images to removes those that don't have a face in them, using the haarcascade_frontalface_default classifier.

    Args:
        images (list[np.ndarray]): the list of images to filter
        min_faces (int, optional): the minimum number of faces required. Defaults to 1.
        max_faces (int, optional): the maximum number of faces required. Defaults to 100.

    Returns:
        list[np.ndarray]: list of filtered images
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    result = []

    for image in images:
        image_PIL = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(image_PIL, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        face_count = len(faces)
        if min_faces <= face_count:
            if max_faces is None or face_count <= max_faces:
                result.append(image)
    return result


def filter_by_similarity(
    images: list[np.ndarray], threshold: float = 0.98
) -> list[np.ndarray]:
    """
    Filters images based on their similarity, using CLIP embeddings and cosine similarity.
    Only the most distinct images (based on similarity threshold) are kept.

    Args:
        images (list[np.ndarray]): the list of images to filter.
        threshold (float, optional): the similarity threshold to filter images. Defaults to 0.98.
        max_size (int, optional): the maximum dimension size to which images will be resized. Defaults to 512.

    Returns:
        list[np.ndarray]: list of filtered images.
    """

    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    def compute_embedding(image: np.ndarray) -> np.ndarray:
        """
        Computes the CLIP embedding of an image and returns a normalized embedding for easier comparison.
        """
        try:
            # conversion is needed from the numpy [0,1] images and the Pillow [0, 255] images.
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    image = np.uint8(np.clip(image * 255, 0, 255))

            pil_image = Image.fromarray(image)
            image_input = preprocess(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                embedding = model.encode_image(image_input).cpu().numpy()
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error embedding image:\n{e}")
            return None

    def are_similar(
        embedding1: np.ndarray, embedding2: np.ndarray, threshold: float
    ) -> bool:
        """
        Checks if two images are similar (if the dot product of their CLIP embeddings is above a threshold).
        """
        embedding1 = np.squeeze(embedding1)
        embedding2 = np.squeeze(embedding2)
        # using dot product of the embedding to comput similarity
        similarity = np.dot(embedding1, embedding2)
        return similarity >= threshold

    embeddings = {}
    with yaspin(
        Spinners.dots,
        text="Computing CLIP embeddings of the images",
    ) as clip_sp:
        for idx, img in enumerate(images):
            embeddings[idx] = compute_embedding(img)
    clip_sp.ok("✓")

    result = []  # this will be the final result
    processed_indices = set()  # indices of images similar to another image

    with yaspin(
        Spinners.dots,
        text="Comparing all CLIP embeddings and filtering similar images",
    ) as embeddings_comparison_sp:
        for i, image1 in enumerate(images):
            if i in processed_indices:
                continue  # skip this image since it already has been checked
            embedding1 = embeddings.get(i)
            if embedding1 is None:
                continue  # ignore images that couldn't be embedded
            similar_images = [(i, image1)]
            for j, image2 in enumerate(images):
                if i != j and j not in processed_indices:
                    embedding2 = embeddings.get(j)
                    if embedding2 is None:
                        continue
                    if are_similar(embedding1, embedding2, threshold):
                        similar_images.append((j, image2))
            # when all images similar to image1 have been found, only keep the one with the highest resolution.
            best_image = max(similar_images, key=lambda x: x[1].size)
            result.append(best_image[1])
            # mark all similar images as processed to avoid repetition
            for index, _ in similar_images:
                processed_indices.add(index)
    embeddings_comparison_sp.ok("✓")
    return result


def sort_by_aesthetic(images: list[np.ndarray], batch_size: int = 1) -> list[float]:
    """
    Sorts a list of images according to an aesthetic score. This aesthetic score is estimated by the aesthetic predictor v2.5.

    Args:
        images (list[np.ndarray]): the list of the images that need to be sorted.
        batch_size (int): size of batches for model inference. Bigger batch size means more VRAM used, but can improve the inference speed. If len(images) is not divisible by the batch size, the remaining images will be processed together in a smaller batch. Defaults to 1 (no batched processing).

    Returns:
        list[np.ndarray]: list of images sorted by aesthetic score.
    """
    if not len(images):
        return []

    def run_inference_batched(batch: list, model, preprocessor) -> list[float]:
        inputs = preprocessor(batch, return_tensors="pt")
        inputs = inputs.pixel_values.to(torch.bfloat16).to(DEVICE)
        with torch.inference_mode():
            outputs = model(inputs).logits.squeeze().float().cpu().numpy()
        return list(outputs)

    # convert all images to PIL format
    images_pil = [np_to_pil(image) for image in images]

    # load the model once
    model, preprocess = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to(DEVICE)

    if len(images_pil) < batch_size:
        # only a single batch needs to be processed
        scores = run_inference_batched(images_pil, model, preprocess)
    else:
        # compute scores batch by batch
        scores = []
        for i in range(0, len(images_pil) // batch_size - 1):
            batch = images_pil[i * batch_size : i * batch_size + batch_size]
            scores += run_inference_batched(batch, model, preprocess)
        if len(images_pil) % batch_size != 0:
            inputs = images_pil[(i + 1) * batch_size + batch_size :]  # remaining images
            scores += run_inference_batched(inputs, model, preprocess)

    scores = map(float, scores)
    sorted_images = [
        image
        for image, _ in sorted(zip(images, scores), key=lambda x: x[1], reverse=True)
    ]
    return sorted_images


def save_images(
    original_directory: str, images: list[np.ndarray], appendix: str = "_filtered"
) -> str:
    """
    Save a list of a images next to another directory, copying the name of the original directory and appending a keyword. This function assumes the images are not saved yet.

    Args:
        original_directory (str): path to the original directory, next to which the images will be saved. The name of this directory will be used to name the created directory.
        images (list[np.ndarray]): list of all the images that need to be saved.
        appendix (str): this string will be appended to the name of the original directory to create the new directory name. Cannot be empty. Defaults to "filtered".

    Returns:
        str: path of the newly created directory, containing the images.
    """
    parent_dir, original_name = os.path.split(original_directory.rstrip(os.sep + "/"))
    filtered_directory = os.path.join(parent_dir, f"{original_name}{appendix}")
    os.makedirs(filtered_directory, exist_ok=True)
    for i, image in enumerate(images):
        image_to_save = (image * 255).astype(np.uint8)
        file_name = f"img{i:03d}.png"
        file_path = os.path.join(filtered_directory, file_name)
        cv2.imwrite(file_path, image_to_save)

    return filtered_directory


if __name__ == "__main__":
    args = get_args()

    with yaspin(Spinners.dots, text="Loading images") as loading_sp:
        raw_images = load_images(args.path)
    loading_sp.ok("✓")

    with yaspin(Spinners.dots, text="Filtering images by resolution") as resolution_sp:
        images = filter_by_resolution(
            raw_images, [(1024, 1024), (1344, 768), (928, 1152)]
        )
    resolution_sp.ok("✓")

    if 0 < args.min_faces or not args.max_faces is None:
        with yaspin(
            Spinners.dots,
            text="Filtering images by faces",
        ) as faces_sp:
            images = filter_by_faces(images, args.min_faces, args.max_faces)
        faces_sp.ok("✓")

    if not args.keep_similar:
        images = filter_by_similarity(images)

    if not args.ignore_aesthetic:
        with yaspin(
            Spinners.dots, text="Sorting images by aesthetic score"
        ) as aesthetic_sp:
            images = sort_by_aesthetic(images, batch_size=64)
        aesthetic_sp.ok("✓")

    with yaspin(Spinners.dots, text="Saving filtered images, DOT NOT EXIT!") as save_sp:
        filtered_dataset = save_images(args.path, images)
    save_sp.ok("✓")

    print(
        f"Filtre is done, went from {len(raw_images)} to {len(images)} images ({int((len(raw_images)-len(images))/len(raw_images)*100)}% reduction). Filtered dataset has been saved in {filtered_dataset}."
    )
