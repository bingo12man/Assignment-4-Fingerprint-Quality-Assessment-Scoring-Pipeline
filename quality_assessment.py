from pathlib import Path
import cv2
import numpy as np

# Provisional value. It will be recalibrated using the complete test set.
DEFAULT_BLUR_THRESHOLD = 8.5

DEFAULT_GLARE_PIXEL_THRESHOLD = 240
DEFAULT_GLARE_FRACTION_THRESHOLD = 0.05

DEFAULT_ROI_FRACTION_THRESHOLD = 0.15

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

def check_brightness(
        image_bgr: np.ndarray,
        dark_threshold: float = 50.0,
        bright_threshold: float = 210.0,
) -> dict:

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The input image must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    if dark_threshold >= bright_threshold:
        raise ValueError(
            "Dark threshold must be lower than bright threshold."
        )

    image_gray = convert_to_grayscale(image_bgr)

    brightness = float(image_gray.mean())

    dark_pixel_fraction = float(
        np.mean(image_gray < dark_threshold)
    )

    bright_pixel_fraction = float(
        np.mean(image_gray > bright_threshold)
    )

    too_dark = brightness < dark_threshold
    too_bright = brightness > bright_threshold

    return {
        "brightness": round(brightness, 2),
        "dark_threshold": float(dark_threshold),
        "bright_threshold": float(bright_threshold),
        "dark_pixel_fraction": round(dark_pixel_fraction, 4),
        "bright_pixel_fraction": round(bright_pixel_fraction, 4),
        "too_dark": bool(too_dark),
        "too_bright": bool(too_bright),
        "passed": bool(not too_dark and not too_bright),
    }

def check_glare(
    image_bgr: np.ndarray,
    pixel_threshold: int = DEFAULT_GLARE_PIXEL_THRESHOLD,
    fraction_threshold: float = DEFAULT_GLARE_FRACTION_THRESHOLD,
) -> dict:

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The input image must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    if not 0 <= pixel_threshold <= 255:
        raise ValueError("Pixel threshold must be between 0 and 255.")

    if not 0.0 <= fraction_threshold <= 1.0:
        raise ValueError(
            "Fraction threshold must be between 0.0 and 1.0."
        )

    image_gray = convert_to_grayscale(image_bgr)

    glare_mask = image_gray > pixel_threshold

    glare_pixel_count = int(np.count_nonzero(glare_mask))
    total_pixel_count = int(glare_mask.size)

    glare_fraction = glare_pixel_count / total_pixel_count
    has_glare = glare_fraction > fraction_threshold

    return {
        "glare_fraction": round(float(glare_fraction), 4),
        "glare_percentage": round(float(glare_fraction * 100), 2),
        "glare_pixel_count": glare_pixel_count,
        "total_pixel_count": total_pixel_count,
        "maximum_intensity": int(image_gray.max()),
        "pixel_threshold": int(pixel_threshold),
        "fraction_threshold": float(fraction_threshold),
        "has_glare": bool(has_glare),
        "passed": bool(not has_glare),
    }


def create_finger_mask(image_bgr: np.ndarray) -> np.ndarray:

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("The input image must be a NumPy array.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    image_gray = convert_to_grayscale(image_bgr)

    blurred_gray = cv2.GaussianBlur(
        image_gray,
        (5, 5),
        sigmaX=0,
    )

    height, width = blurred_gray.shape

    border_size = max(1, min(height, width) // 20)

    border_pixels = np.concatenate(
        [
            blurred_gray[:border_size, :].ravel(),
            blurred_gray[-border_size:, :].ravel(),
            blurred_gray[:, :border_size].ravel(),
            blurred_gray[:, -border_size:].ravel(),
        ]
    )

    border_mean = float(border_pixels.mean())
    image_mean = float(blurred_gray.mean())

    if border_mean >= image_mean:
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        threshold_type = cv2.THRESH_BINARY

    _, raw_mask = cv2.threshold(
        blurred_gray,
        0,
        255,
        threshold_type | cv2.THRESH_OTSU,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    cleaned_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    final_mask = np.zeros_like(cleaned_mask)

    if not contours:
        return final_mask

    image_center = (
        width / 2.0,
        height / 2.0,
    )

    central_contours = [
        contour
        for contour in contours
        if cv2.pointPolygonTest(
            contour,
            image_center,
            False,
        )
        >= 0
    ]

    candidate_contours = (
        central_contours if central_contours else contours
    )

    selected_contour = max(
        candidate_contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        final_mask,
        [selected_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    foreground_fraction = (
        np.count_nonzero(final_mask) / final_mask.size
    )

    # A single finger normally should not occupy most of the frame.
    # If more than half the frame is white, the selected region is
    # probably the background, so invert the mask.
    if foreground_fraction > 0.50:
        final_mask = cv2.bitwise_not(final_mask)

    # Keep only the largest connected foreground region after inversion.
    final_contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    clean_final_mask = np.zeros_like(final_mask)

    if not final_contours:
        return clean_final_mask

    finger_contour = max(
        final_contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        clean_final_mask,
        [finger_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return clean_final_mask

def check_roi_completeness(
    image_bgr: np.ndarray,
    threshold: float = DEFAULT_ROI_FRACTION_THRESHOLD,
) -> dict:

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("ROI threshold must be between 0.0 and 1.0.")

    finger_mask = create_finger_mask(image_bgr)

    roi_pixel_count = int(np.count_nonzero(finger_mask))
    total_pixel_count = int(finger_mask.size)

    roi_fraction = roi_pixel_count / total_pixel_count
    roi_complete = roi_fraction >= threshold

    contours, _ = cv2.findContours(
        finger_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    bounding_box = None

    if contours:
        selected_contour = max(
            contours,
            key=cv2.contourArea,
        )

        x, y, width, height = cv2.boundingRect(
            selected_contour
        )

        bounding_box = {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        }

    border_contact = {
        "top": False,
        "bottom": False,
        "left": False,
        "right": False,
    }

    unexpected_border_touch = False

    if bounding_box is not None:
        x = bounding_box["x"]
        y = bounding_box["y"]
        width = bounding_box["width"]
        height = bounding_box["height"]

        image_height, image_width = finger_mask.shape

        border_contact = {
            "top": y <= 0,
            "bottom": y + height >= image_height,
            "left": x <= 0,
            "right": x + width >= image_width,
        }

        # Bottom contact is expected because the finger enters from below.
        unexpected_border_touch = (
            border_contact["top"]
            or border_contact["left"]
            or border_contact["right"]
        )
    return {
        "roi_fraction": round(float(roi_fraction), 4),
        "roi_percentage": round(float(roi_fraction * 100), 2),
        "roi_pixel_count": roi_pixel_count,
        "total_pixel_count": total_pixel_count,
        "threshold": float(threshold),
        "roi_complete": bool(roi_complete),
        "passed": bool(roi_complete),
        "bounding_box": bounding_box,
        "border_contact": border_contact,
        "unexpected_border_touch": bool(unexpected_border_touch),
    }

def save_roi_visualizations(
    image_bgr: np.ndarray,
    mask_path: str,
    overlay_path: str,
) -> None:

    finger_mask = create_finger_mask(image_bgr)

    if not cv2.imwrite(mask_path, finger_mask):
        raise IOError(f"Could not save ROI mask: {mask_path}")

    overlay = image_bgr.copy()

    contours, _ = cv2.findContours(
        finger_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        selected_contour = max(
            contours,
            key=cv2.contourArea,
        )

        cv2.drawContours(
            overlay,
            [selected_contour],
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=8,
        )

        x, y, width, height = cv2.boundingRect(
            selected_contour
        )

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            color=(255, 0, 0),
            thickness=8,
        )

    if not cv2.imwrite(overlay_path, overlay):
        raise IOError(f"Could not save ROI overlay: {overlay_path}")

    
def main() -> None:
    """Compare ROI completeness for normal and distant captures."""

    Path("outputs").mkdir(exist_ok=True)

    test_cases = [
        (
            "Good image",
            "data/good/good_01.png",
            "outputs/good_01_roi_mask.png",
            "outputs/good_01_roi_overlay.png",
        ),
        (
            "Finger too small",
            "data/roi/too_small_01.png",
            "outputs/too_small_01_roi_mask.png",
            "outputs/too_small_01_roi_overlay.png",
        ),
    ]

    for label, image_path, mask_path, overlay_path in test_cases:
        image_bgr = load_image(image_path)

        result = check_roi_completeness(image_bgr)

        save_roi_visualizations(
            image_bgr,
            mask_path,
            overlay_path,
        )

        print(f"\n{label}")
        print(f"Path: {image_path}")
        print(f"ROI percentage: {result['roi_percentage']}%")
        print(f"Required percentage: {result['threshold'] * 100:.2f}%")
        print(f"ROI complete: {result['roi_complete']}")
        print(f"Bounding box: {result['bounding_box']}")
        print(f"Mask: {mask_path}")
        print(f"Overlay: {overlay_path}")
        print(f"Border contact: {result['border_contact']}")
        print(
            f"Unexpected border touch: "
            f"{result['unexpected_border_touch']}"
        )

if __name__ == "__main__":
    main()