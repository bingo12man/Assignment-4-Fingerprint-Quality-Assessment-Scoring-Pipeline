from pathlib import Path
import cv2
import numpy as np

# Provisional value. It will be recalibrated using the complete test set.
DEFAULT_BLUR_THRESHOLD = 8.5

DEFAULT_GLARE_PIXEL_THRESHOLD = 240
DEFAULT_GLARE_FRACTION_THRESHOLD = 0.05

DEFAULT_ROI_FRACTION_THRESHOLD = 0.15

DEFAULT_GABOR_KERNEL_SIZE = 31
DEFAULT_GABOR_SIGMA = 4.0
DEFAULT_GABOR_WAVELENGTH = 10.0
DEFAULT_GABOR_GAMMA = 0.5
DEFAULT_GABOR_ORIENTATIONS = 6

# Provisional threshold; recalibrate using the complete image dataset.
DEFAULT_RIDGE_CLARITY_THRESHOLD = 1.5

DEFAULT_QUALITY_WEIGHTS = {
    "blur": 0.25,
    "brightness": 0.15,
    "glare": 0.10,
    "roi": 0.20,
    "ridge": 0.30,
}

METRIC_ORDER = (
    "blur",
    "brightness",
    "glare",
    "roi",
    "ridge",
)

DEFAULT_COMPOSITE_THRESHOLD = 60.0


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

def extract_fingertip_patch(
    image_bgr: np.ndarray,
    finger_mask: np.ndarray,
    height_ratio: float = 0.35,
    padding_ratio: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")

    if finger_mask.ndim != 2:
        raise ValueError("Expected a single-channel finger mask.")

    if image_bgr.shape[:2] != finger_mask.shape:
        raise ValueError(
            "Image and finger mask must have matching dimensions."
        )

    if not 0.0 < height_ratio <= 1.0:
        raise ValueError("Height ratio must be between 0 and 1.")

    foreground_y, foreground_x = np.where(finger_mask > 0)

    if foreground_x.size == 0:
        raise ValueError("Finger mask contains no foreground pixels.")

    top_y = int(foreground_y.min())
    bottom_y = int(foreground_y.max())

    finger_height = bottom_y - top_y + 1

    patch_height = max(
        1,
        int(round(finger_height * height_ratio)),
    )

    patch_bottom = min(
        finger_mask.shape[0],
        top_y + patch_height,
    )

    top_region_mask = finger_mask[
        top_y:patch_bottom,
        :
    ]

    # Break narrow connections between the fingertip and shadows/background.
    region_height, region_width = top_region_mask.shape

    kernel_size = max(
        3,
        min(region_height, region_width) // 60,
    )

    # Morphological kernels should use odd dimensions.
    if kernel_size % 2 == 0:
        kernel_size += 1

    separation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    separated_mask = cv2.morphologyEx(
        top_region_mask,
        cv2.MORPH_OPEN,
        separation_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        separated_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(
            "No foreground component found in the fingertip region."
        )

    minimum_area = top_region_mask.size * 0.005

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= minimum_area
    ]

    if not valid_contours:
        raise ValueError(
            "No sufficiently large fingertip component was found."
        )

    image_center_x = finger_mask.shape[1] / 2.0

    def contour_score(contour: np.ndarray) -> float:
        """
        Prefer a large contour located near the horizontal centre.
        """

        area = cv2.contourArea(contour)

        x, _, width, _ = cv2.boundingRect(contour)
        contour_center_x = x + width / 2.0

        normalized_distance = (
            abs(contour_center_x - image_center_x)
            / finger_mask.shape[1]
        )

        return area / (1.0 + 4.0 * normalized_distance)

    selected_contour = max(
        valid_contours,
        key=contour_score,
    )

    component_seed = np.zeros_like(top_region_mask)

    cv2.drawContours(
        component_seed,
        [selected_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    # Recover some area removed during opening.
    recovered_component = cv2.dilate(
        component_seed,
        separation_kernel,
        iterations=2,
    )

    # Keep only pixels that belonged to the original mask.
    isolated_top_mask = cv2.bitwise_and(
        recovered_component,
        top_region_mask,
    )

    x, y, width, height = cv2.boundingRect(
        selected_contour
    )

    padding_x = int(round(width * padding_ratio))
    padding_y = int(round(height * padding_ratio))

    left_x = max(0, x - padding_x)
    right_x = min(
        isolated_top_mask.shape[1],
        x + width + padding_x,
    )

    upper_y = max(0, y - padding_y)
    lower_y = min(
        isolated_top_mask.shape[0],
        y + height + padding_y,
    )

    image_gray = convert_to_grayscale(image_bgr)

    top_region_gray = image_gray[
        top_y:patch_bottom,
        :
    ]

    fingertip_patch = top_region_gray[
        upper_y:lower_y,
        left_x:right_x,
    ]

    fingertip_mask = isolated_top_mask[
        upper_y:lower_y,
        left_x:right_x,
    ]

    return fingertip_patch, fingertip_mask

def create_ridge_core_mask(
    fingertip_mask: np.ndarray,
    core_ratio: float = 0.35,
) -> np.ndarray:

    if fingertip_mask.ndim != 2:
        raise ValueError(
            "Expected a single-channel fingertip mask."
        )

    if not 0.0 < core_ratio < 1.0:
        raise ValueError(
            "Core ratio must be between 0 and 1."
        )

    binary_mask = (
        fingertip_mask > 0
    ).astype(np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        raise ValueError(
            "Fingertip mask contains no foreground pixels."
        )

    distance_map = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        maskSize=5,
    )

    _, maximum_distance, _, maximum_location = cv2.minMaxLoc(
        distance_map
    )

    if maximum_distance <= 0:
        raise ValueError(
            "Could not calculate a valid fingertip distance map."
        )

    minimum_core_distance = (
        maximum_distance * core_ratio
    )

    initial_core = (
        distance_map >= minimum_core_distance
    ).astype(np.uint8)

    number_of_labels, labels, _, _ = (
        cv2.connectedComponentsWithStats(
            initial_core,
            connectivity=8,
        )
    )

    maximum_x, maximum_y = maximum_location
    selected_label = labels[maximum_y, maximum_x]

    if selected_label == 0 or number_of_labels <= 1:
        raise ValueError(
            "Could not isolate the central fingertip region."
        )

    core_mask = np.where(
        labels == selected_label,
        255,
        0,
    ).astype(np.uint8)

    return core_mask

def save_ridge_core(
    image_bgr: np.ndarray,
    output_path: str,
) -> None:

    finger_mask = create_finger_mask(image_bgr)

    fingertip_patch, fingertip_mask = extract_fingertip_patch(
        image_bgr,
        finger_mask,
    )

    ridge_core_mask = create_ridge_core_mask(
        fingertip_mask
    )

    visualization = np.full_like(
        fingertip_patch,
        fill_value=127,
    )

    visualization[ridge_core_mask > 0] = (
        fingertip_patch[ridge_core_mask > 0]
    )

    if not cv2.imwrite(output_path, visualization):
        raise IOError(
            f"Could not save ridge-core image: {output_path}"
        )
    

def save_fingertip_patch(
    image_bgr: np.ndarray,
    output_path: str,
) -> None:

    finger_mask = create_finger_mask(image_bgr)

    fingertip_patch, fingertip_mask = extract_fingertip_patch(
        image_bgr,
        finger_mask,
    )

    masked_patch = np.full_like(
        fingertip_patch,
        fill_value=127,
    )

    masked_patch[fingertip_mask > 0] = (
        fingertip_patch[fingertip_mask > 0]
    )

    if not cv2.imwrite(output_path, masked_patch):
        raise IOError(
            f"Could not save fingertip patch: {output_path}"
        )

def prepare_ridge_patch(
    fingertip_patch: np.ndarray,
    ridge_core_mask: np.ndarray,
    max_dimension: int = 512,
) -> tuple[np.ndarray, np.ndarray]:

    if fingertip_patch.ndim != 2:
        raise ValueError(
            "Expected a single-channel fingertip patch."
        )

    if ridge_core_mask.ndim != 2:
        raise ValueError(
            "Expected a single-channel ridge-core mask."
        )

    if fingertip_patch.shape != ridge_core_mask.shape:
        raise ValueError(
            "Patch and ridge-core mask must have matching dimensions."
        )

    if max_dimension <= 0:
        raise ValueError(
            "Maximum dimension must be positive."
        )

    height, width = fingertip_patch.shape
    largest_dimension = max(height, width)

    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension

        resized_width = max(
            1,
            int(round(width * scale)),
        )
        resized_height = max(
            1,
            int(round(height * scale)),
        )

        fingertip_patch = cv2.resize(
            fingertip_patch,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        ridge_core_mask = cv2.resize(
            ridge_core_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_patch = clahe.apply(fingertip_patch)

    ridge_core_mask = np.where(
        ridge_core_mask > 0,
        255,
        0,
    ).astype(np.uint8)

    return enhanced_patch, ridge_core_mask

def create_gabor_kernels(
    kernel_size: int = DEFAULT_GABOR_KERNEL_SIZE,
    sigma: float = DEFAULT_GABOR_SIGMA,
    wavelength: float = DEFAULT_GABOR_WAVELENGTH,
    gamma: float = DEFAULT_GABOR_GAMMA,
    orientation_count: int = DEFAULT_GABOR_ORIENTATIONS,
) -> list[np.ndarray]:

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Gabor kernel size must be a positive odd number."
        )

    if sigma <= 0 or wavelength <= 0 or gamma <= 0:
        raise ValueError(
            "Sigma, wavelength and gamma must be positive."
        )

    if orientation_count <= 0:
        raise ValueError(
            "Orientation count must be positive."
        )

    kernels = []

    for orientation_index in range(orientation_count):
        theta = (
            orientation_index
            * np.pi
            / orientation_count
        )

        kernel = cv2.getGaborKernel(
            ksize=(kernel_size, kernel_size),
            sigma=sigma,
            theta=theta,
            lambd=wavelength,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F,
        )

        # Remove the DC component so uniform brightness produces
        # less response than periodic ridge texture.
        kernel = kernel - kernel.mean()

        kernel_norm = float(
            np.sum(np.abs(kernel))
        )

        if kernel_norm > 0:
            kernel = kernel / kernel_norm

        kernels.append(kernel.astype(np.float32))

    return kernels

def calculate_gabor_response(
    enhanced_patch: np.ndarray,
    kernels: list[np.ndarray],
) -> np.ndarray:

    if enhanced_patch.ndim != 2:
        raise ValueError(
            "Expected a single-channel enhanced patch."
        )

    if not kernels:
        raise ValueError(
            "At least one Gabor kernel is required."
        )

    patch_float = enhanced_patch.astype(
        np.float32
    )

    maximum_response = np.zeros(
        enhanced_patch.shape,
        dtype=np.float32,
    )

    for kernel in kernels:
        response = cv2.filter2D(
            patch_float,
            ddepth=cv2.CV_32F,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT,
        )

        absolute_response = np.abs(response)

        maximum_response = np.maximum(
            maximum_response,
            absolute_response,
        )

    return maximum_response

def check_ridge_clarity(
    image_bgr: np.ndarray,
    threshold: float | None = DEFAULT_RIDGE_CLARITY_THRESHOLD,
) -> dict:

    finger_mask = create_finger_mask(image_bgr)

    fingertip_patch, fingertip_mask = extract_fingertip_patch(
        image_bgr,
        finger_mask,
    )

    ridge_core_mask = create_ridge_core_mask(
        fingertip_mask
    )

    enhanced_patch, ridge_core_mask = prepare_ridge_patch(
        fingertip_patch,
        ridge_core_mask,
    )

    kernels = create_gabor_kernels()

    response_map = calculate_gabor_response(
        enhanced_patch,
        kernels,
    )

    # Remove mask-boundary pixels so the Gabor score measures
    # interior ridges rather than the edge of the finger mask.
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    interior_mask = cv2.erode(
        ridge_core_mask,
        erosion_kernel,
        iterations=1,
    )

    if np.count_nonzero(interior_mask) == 0:
        interior_mask = ridge_core_mask

    valid_responses = response_map[
        interior_mask > 0
    ]

    if valid_responses.size == 0:
        raise ValueError(
            "No valid ridge-response pixels were found."
        )

    ridge_score = float(
        np.var(valid_responses)
    )

    mean_response = float(
        np.mean(valid_responses)
    )

    passed = None

    if threshold is not None:
        if threshold < 0:
            raise ValueError(
                "Ridge threshold cannot be negative."
            )

        passed = ridge_score >= threshold

    return {
        "ridge_score": round(ridge_score, 4),
        "mean_gabor_response": round(
            mean_response,
            4,
        ),
        "threshold": (
            None
            if threshold is None
            else float(threshold)
        ),
        "passed": (
            None
            if passed is None
            else bool(passed)
        ),
        "processed_shape": enhanced_patch.shape,
        "orientation_count": len(kernels),
        "response_map": response_map,
        "analysis_mask": interior_mask,
    }

def save_gabor_response(
    result: dict,
    output_path: str,
) -> None:

    response_map = result["response_map"]
    analysis_mask = result["analysis_mask"]

    visible_response = np.zeros(
        response_map.shape,
        dtype=np.uint8,
    )

    valid_values = response_map[
        analysis_mask > 0
    ]

    if valid_values.size == 0:
        raise ValueError(
            "No valid Gabor responses are available."
        )

    minimum_value = float(valid_values.min())
    maximum_value = float(valid_values.max())

    if maximum_value > minimum_value:
        normalized = (
            (response_map - minimum_value)
            / (maximum_value - minimum_value)
            * 255.0
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        visible_response[
            analysis_mask > 0
        ] = normalized[
            analysis_mask > 0
        ]

    if not cv2.imwrite(
        output_path,
        visible_response,
    ):
        raise IOError(
            f"Could not save Gabor response: {output_path}"
        )

    
# def save_roi_visualizations(
#     image_bgr: np.ndarray,
#     mask_path: str,
#     overlay_path: str,
# ) -> None:

#     finger_mask = create_finger_mask(image_bgr)

#     if not cv2.imwrite(mask_path, finger_mask):
#         raise IOError(f"Could not save ROI mask: {mask_path}")

#     overlay = image_bgr.copy()

#     contours, _ = cv2.findContours(
#         finger_mask,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE,
#     )

#     if contours:
#         selected_contour = max(
#             contours,
#             key=cv2.contourArea,
#         )

#         cv2.drawContours(
#             overlay,
#             [selected_contour],
#             contourIdx=-1,
#             color=(0, 255, 0),
#             thickness=8,
#         )

#         x, y, width, height = cv2.boundingRect(
#             selected_contour
#         )

#         cv2.rectangle(
#             overlay,
#             (x, y),
#             (x + width, y + height),
#             color=(255, 0, 0),
#             thickness=8,
#         )

#     if not cv2.imwrite(overlay_path, overlay):
#         raise IOError(f"Could not save ROI overlay: {overlay_path}")

def clamp_01(value: float) -> float:

    return max(
        0.0,
        min(1.0, float(value)),
    )

def normalize_higher_is_better(
    value: float,
    threshold: float,
) -> float:

    if threshold <= 0:
        raise ValueError(
            "Normalization threshold must be positive."
        )

    target_value = threshold * 2.0

    normalized = value / target_value

    return clamp_01(normalized)

def normalize_brightness(
    brightness: float,
    dark_threshold: float,
    bright_threshold: float,
    ideal_low: float = 90.0,
    ideal_high: float = 180.0,
) -> float:

    if not (
        dark_threshold
        < ideal_low
        <= ideal_high
        < bright_threshold
    ):
        raise ValueError(
            "Brightness ranges must satisfy: "
            "dark < ideal_low <= ideal_high < bright."
        )

    if brightness <= dark_threshold:
        return 0.0

    if brightness >= bright_threshold:
        return 0.0

    if ideal_low <= brightness <= ideal_high:
        return 1.0

    if brightness < ideal_low:
        normalized = (
            (brightness - dark_threshold)
            / (ideal_low - dark_threshold)
        )

        return clamp_01(normalized)

    normalized = (
        (bright_threshold - brightness)
        / (bright_threshold - ideal_high)
    )

    return clamp_01(normalized)

def normalize_glare(
    glare_fraction: float,
    fraction_threshold: float,
) -> float:

    if fraction_threshold <= 0:
        raise ValueError(
            "Glare fraction threshold must be positive."
        )

    normalized = (
        1.0
        - glare_fraction / fraction_threshold
    )

    return clamp_01(normalized)

def normalize_quality_metrics(
    blur_result: dict,
    brightness_result: dict,
    glare_result: dict,
    roi_result: dict,
    ridge_result: dict,
) -> dict:
    
    blur_normalized = normalize_higher_is_better(
        blur_result["blur_score"],
        blur_result["threshold"],
    )

    brightness_normalized = normalize_brightness(
        brightness_result["brightness"],
        brightness_result["dark_threshold"],
        brightness_result["bright_threshold"],
    )

    glare_normalized = normalize_glare(
        glare_result["glare_fraction"],
        glare_result["fraction_threshold"],
    )

    roi_normalized = normalize_higher_is_better(
        roi_result["roi_fraction"],
        roi_result["threshold"],
    )

    ridge_normalized = normalize_higher_is_better(
        ridge_result["ridge_score"],
        ridge_result["threshold"],
    )

    return {
        "blur": round(blur_normalized, 4),
        "brightness": round(brightness_normalized, 4),
        "glare": round(glare_normalized, 4),
        "roi": round(roi_normalized, 4),
        "ridge": round(ridge_normalized, 4),
    }

def calculate_composite_score(
    normalized_metrics: dict,
    weights: dict | None = None,
    pass_threshold: float = DEFAULT_COMPOSITE_THRESHOLD,
) -> dict:

    if weights is None:
        weights = DEFAULT_QUALITY_WEIGHTS.copy()

    required_metrics = set(METRIC_ORDER)

    if set(normalized_metrics) != required_metrics:
        raise ValueError(
            "Normalized metrics must contain exactly: "
            "blur, brightness, glare, roi and ridge."
        )

    if set(weights) != required_metrics:
        raise ValueError(
            "Weights must contain exactly: "
            "blur, brightness, glare, roi and ridge."
        )

    total_weight = sum(weights.values())

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"Quality weights must total 1.0, got {total_weight:.4f}."
        )

    if not 0.0 <= pass_threshold <= 100.0:
        raise ValueError(
            "Composite threshold must be between 0 and 100."
        )

    weighted_contributions = {}

    for metric_name in METRIC_ORDER:
        normalized_value = float(
            normalized_metrics[metric_name]
        )

        if not 0.0 <= normalized_value <= 1.0:
            raise ValueError(
                f"Normalized {metric_name} score must be "
                "between 0 and 1."
            )

        weight = float(weights[metric_name])

        if weight < 0:
            raise ValueError(
                f"Weight for {metric_name} cannot be negative."
            )

        contribution = (
            normalized_value
            * weight
            * 100.0
        )

        weighted_contributions[metric_name] = round(
            contribution,
            2,
        )

    composite_score = sum(
        normalized_metrics[metric_name]
        * weights[metric_name]
        for metric_name in required_metrics
    ) * 100.0

    passed = composite_score >= pass_threshold

    return {
        "composite_score": round(
            float(composite_score),
            2,
        ),
        "pass_threshold": float(pass_threshold),
        "passed": bool(passed),
        "weights": weights,
        "contributions": weighted_contributions,
    }

def generate_quality_guidance(
    blur_result: dict,
    brightness_result: dict,
    glare_result: dict,
    roi_result: dict,
    ridge_result: dict,
) -> tuple[str, list[str]]:

    issues = []

    if not roi_result["roi_complete"]:
        issues.append(
            "Finger is too small — move it closer to the camera."
        )

    if brightness_result["too_dark"]:
        issues.append(
            "Image is too dark — increase the lighting and retry."
        )

    if brightness_result["too_bright"]:
        issues.append(
            "Image is too bright — reduce the lighting and retry."
        )

    if glare_result["has_glare"]:
        issues.append(
            "Glare detected — change the light angle and retry."
        )

    if blur_result["is_blurry"]:
        issues.append(
            "Image is too blurry — hold your hand steady and retry."
        )

    if ridge_result["passed"] is False:
        issues.append(
            "Fingerprint ridges are unclear — adjust focus and distance."
        )

    if not issues:
        return (
            "Good capture — ready for processing",
            [],
        )

    return issues[0], issues

def quality_gate(
    image_path: str,
) -> dict:
    
    image_bgr = load_image(image_path)

    blur_result = check_blur(image_bgr)
    brightness_result = check_brightness(image_bgr)
    glare_result = check_glare(image_bgr)
    roi_result = check_roi_completeness(image_bgr)
    ridge_result = check_ridge_clarity(image_bgr)

    normalized_metrics = normalize_quality_metrics(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result,
    )

    composite_result = calculate_composite_score(
        normalized_metrics
    )

    guidance, issues = generate_quality_guidance(
        blur_result,
        brightness_result,
        glare_result,
        roi_result,
        ridge_result,
    )

    individual_checks = {
        "blur": not blur_result["is_blurry"],
        "brightness": brightness_result["passed"],
        "glare": glare_result["passed"],
        "roi": roi_result["passed"],
        "ridge": ridge_result["passed"],
    }

    all_individual_checks_passed = all(
        individual_checks.values()
    )

    # Strict quality-gate decision:
    # A serious individual defect cannot be hidden by a high
    # weighted score from the other metrics.
    passed = (
        composite_result["passed"]
        and all_individual_checks_passed
    )

    # Response maps are NumPy arrays used only for visualization.
    # They should not be returned in the public result dictionary.
    ridge_summary = {
        key: value
        for key, value in ridge_result.items()
        if key not in {
            "response_map",
            "analysis_mask",
        }
    }

    return {
        "passed": bool(passed),
        "composite_score": composite_result[
            "composite_score"
        ],
        "composite_threshold": composite_result[
            "pass_threshold"
        ],
        "composite_passed": composite_result[
            "passed"
        ],
        "all_individual_checks_passed": bool(
            all_individual_checks_passed
        ),
        "individual_checks": individual_checks,
        "guidance": guidance,
        "issues": issues,
        "normalized_metrics": normalized_metrics,
        "weighted_contributions": composite_result[
            "contributions"
        ],
        "blur": blur_result,
        "brightness": brightness_result,
        "glare": glare_result,
        "roi": roi_result,
        "ridge": ridge_summary,
    }

def main() -> None:
    """Run the complete quality gate on the good image."""

    image_path = "data/good/good_01.png"

    result = quality_gate(image_path)

    print(f"Image: {image_path}")
    print(
        f"Composite score: "
        f"{result['composite_score']}"
    )
    print(
        f"Required score: "
        f"{result['composite_threshold']}"
    )
    print(
        f"Composite passed: "
        f"{result['composite_passed']}"
    )

    print("\nIndividual checks")

    for metric_name in METRIC_ORDER:
        print(
            f"{metric_name}: "
            f"{result['individual_checks'][metric_name]}"
        )

    print(
        f"\nAll checks passed: "
        f"{result['all_individual_checks_passed']}"
    )
    print(f"Final decision: {result['passed']}")
    print(f"Guidance: {result['guidance']}")

    if result["issues"]:
        print("\nDetected issues")

        for issue in result["issues"]:
            print(f"- {issue}")


if __name__ == "__main__":
    main()