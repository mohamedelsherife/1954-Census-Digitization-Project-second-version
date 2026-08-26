import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob


def display(im_path_or_array):
    """Displays an image, whether it's a file path or a numpy array."""
    dpi = 80
    if isinstance(im_path_or_array, str):
        im_data = plt.imread(im_path_or_array)
    else:
        im_data = im_path_or_array

    height, width = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.show()


def process_image(
    image_path,
    output_dir="data/processed",
    block_size=31,
    c_value=15,
    min_speckle_size=4,
    show_steps=False,
):
    """
    Cleans a scanned document image and prepares it for reading/extraction:
    1) Convert to grayscale
    2) Adaptive Threshold (local binarization, unaffected by dark edges)
    3) Remove noise (small scattered dots/speckles) via Connected Components

    Parameters:
    - image_path: path to the input image
    - output_dir: folder to save the results
    - block_size: local window size for adaptiveThreshold (odd number)
    - c_value: value subtracted from the mean in adaptiveThreshold
    - min_speckle_size: minimum area (in pixels) considered part of real text
    - show_steps: if True, displays each step in detail

    Returns: path to the final cleaned image (denoised.png)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    if show_steps:
        display(image_path)

    # 1) Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, f"{base_name}_gray.jpg")
    cv2.imwrite(gray_path, gray_image)
    if show_steps:
        display(gray_path)

    # 2) Light smoothing + Adaptive Threshold
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    im_bw = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c_value,
    )
    bw_path = os.path.join(output_dir, f"{base_name}_bw.jpg")
    cv2.imwrite(bw_path, im_bw)
    if show_steps:
        display(bw_path)

    # 3) Remove noise via Connected Components
    _, im_bw_strict = cv2.threshold(im_bw, 127, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(im_bw_strict)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_speckle_size:
            cleaned[labels == i] = 255

    denoised = cv2.bitwise_not(cleaned)

    # Saving as PNG is mandatory (JPEG breaks the binary image and reintroduces noise)
    denoised_path = os.path.join(output_dir, f"{base_name}_denoised.png")
    cv2.imwrite(denoised_path, denoised)
    if show_steps:
        display(denoised_path)

    return denoised_path


def process_folder(
    input_dir,
    output_dir="data/processed",
    pattern="*.jpg",
    **kwargs,
):
    """
    Applies process_image to all images in a given folder.
    Example: process_folder("data/raw", pattern="*.jpg")
    """
    image_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not image_paths:
        print(f"No images matching pattern {pattern} found in {input_dir}")
        return []

    results = []
    for path in image_paths:
        print(f"Processing: {path}")
        try:
            out_path = process_image(path, output_dir=output_dir, **kwargs)
            results.append(out_path)
            print(f"  -> Saved to: {out_path}")
        except Exception as e:
            print(f"  !! Error while processing {path}: {e}")

    return results


if __name__ == "__main__":
    # process folder:
    results = process_folder("data/raw", pattern="*.jpg")

    print("Final image:", results)

    # Example 2: process all images in an entire folder (enable when needed)
    # results = process_folder("data/raw", pattern="*.jpg")
    # print(f"Processed {len(results)} images")