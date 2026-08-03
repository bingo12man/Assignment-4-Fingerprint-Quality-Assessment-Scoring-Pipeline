from pathlib import Path
import cv2
import numpy as np

# Provisional value. It will be recalibrated using the complete test set.
DEFAULT_BLUR_THRESHOLD = 8.5

def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"OpenCV could not read the image: {path}")

    return image


def convert_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The Image must be a Numpy array")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image")

    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return grayscale

def prepare_for_blur_analysis(
    image_bgr: np.ndarray,
    max_dimension: int = 1024,
) -> np.ndarray:

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The input image must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    if max_dimension <= 0:
        raise ValueError("Maximum dimension must be positive.")

    height, width = image_bgr.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension

        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))

        image_bgr = cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    image_gray = convert_to_grayscale(image_bgr)

    denoised_gray = cv2.medianBlur(
        image_gray,
        3,
    )

    return denoised_gray


def check_blur(
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_BLUR_THRESHOLD,
    max_dimension: int = 1024,
) -> dict:

    prepared_gray = prepare_for_blur_analysis(
        image_bgr,
        max_dimension=max_dimension,
    )

    laplacian = cv2.Laplacian(
        prepared_gray,
        cv2.CV_64F,
    )

    blur_score = float(laplacian.var())
    is_blurry = blur_score < threshold

    return {
        "blur_score": round(blur_score, 2),
        "threshold": float(threshold),
        "is_blurry": bool(is_blurry),
        "processed_shape": prepared_gray.shape,
    }


def save_laplacian_visualization(
    image_bgr: np.ndarray,
    output_path: str,
) -> None:
    
    prepared_gray = prepare_for_blur_analysis(image_bgr)

    laplacian = cv2.Laplacian(
        prepared_gray,
        cv2.CV_64F,
    )

    absolute_edges = np.abs(laplacian)

    visible_edges = cv2.normalize(
        absolute_edges,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    ).astype(np.uint8)

    saved = cv2.imwrite(output_path, visible_edges)

    if not saved:
        raise IOError(
            f"Could not save Laplacian visualization: {output_path}"
        )

def create_synthetic_blur(
    image_bgr: np.ndarray,
    output_path: str,
    kernel_size: int = 21,
) -> None:

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    blurred_image = cv2.GaussianBlur(
        image_bgr,
        (kernel_size, kernel_size),
        sigmaX=0,
    )

    saved = cv2.imwrite(output_path, blurred_image)

    if not saved:
        raise IOError(f"Could not save blurry image: {output_path}")
    
def main() -> None:
    """Compare blur scores after standardization and denoising."""

    test_cases = [
        ("Original good image", "data/good/good_01.png"),
        (
            "Synthetic blurry image",
            "data/blurry/synthetic_blurry_01.png",
        ),
        ("Real blurry image", "data/blurry/blurry_01.png"),
    ]

    for label, image_path in test_cases:
        image_bgr = load_image(image_path)
        result = check_blur(image_bgr)

        print(f"\n{label}")
        print(f"Path: {image_path}")
        print(f"Processed shape: {result['processed_shape']}")
        print(f"Blur score: {result['blur_score']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Is blurry: {result['is_blurry']}")


if __name__ == "__main__":
    main()