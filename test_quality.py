"""Tests and evaluates the fingerprint quality pipeline on 20 images."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Any

from quality_assessment import METRIC_ORDER, quality_gate


DATA_ROOT = Path("data")
OUTPUT_CSV = Path("outputs/quality_evaluation.csv")

CATEGORIES = (
    "good",
    "blurry",
    "dark",
    "glare",
)

EXPECTED_DEFECT = {
    "good": None,
    "blurry": "blur",
    "dark": "brightness",
    "glare": "glare",
}

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}


def discover_official_images(
    data_root: Path = DATA_ROOT,
) -> list[tuple[str, Path]]:

    test_cases = []

    for category in CATEGORIES:
        category_folder = data_root / category

        if not category_folder.exists():
            raise AssertionError(
                f"Missing category folder: {category_folder}"
            )

        image_paths = sorted(
            path
            for path in category_folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in ALLOWED_EXTENSIONS
                and path.stem.lower().startswith(
                    f"{category}_"
                )
            )
        )

        if len(image_paths) != 5:
            raise AssertionError(
                f"Expected 5 official {category} images, "
                f"but found {len(image_paths)} in "
                f"{category_folder}."
            )

        for image_path in image_paths:
            test_cases.append(
                (
                    category,
                    image_path,
                )
            )

    return test_cases

def evaluate_test_case(
    category: str,
    image_path: Path,
) -> dict[str, Any]:
    """Run the quality gate and verify the expected defect."""

    start_time = perf_counter()

    expected_metric = EXPECTED_DEFECT[
        category
    ]

    try:
        result = quality_gate(
            str(image_path)
        )

    except (ValueError, OSError) as error:
        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000.0

        return {
            "category": category,
            "filename": image_path.name,
            "expected_defect": (
                expected_metric or "none"
            ),
            "correctly_classified": False,
            "final_passed": False,

            # Composite scores
            "raw_score": "",
            "final_score": "",

            # Blur details
            "blur_score": "",
            "blur_threshold": "",
            "blur_strip_scores": "",
            "blur_passed": "",

            # Brightness details
            "finger_brightness": "",
            "global_brightness": "",
            "dark_pixel_fraction": "",
            "bright_pixel_fraction": "",
            "dark_threshold": "",
            "bright_threshold": "",
            "brightness_passed": "",

            # Glare details
            "glare_fraction": "",
            "glare_fraction_threshold": "",
            "roi_glare_fraction": "",
            "roi_glare_fraction_threshold": "",
            "largest_hotspot_fraction": "",
            "hotspot_fraction_threshold": "",
            "largest_hotspot_pixels": "",
            "global_glare_detected": "",
            "localized_glare_detected": "",
            "glare_passed": "",

            # ROI details
            "roi_fraction": "",
            "roi_threshold": "",
            "roi_passed": "",

            # Ridge details
            "ridge_score": "",
            "ridge_threshold": "",
            "ridge_passed": "",

            # Performance and guidance
            "processing_time_ms": round(
                elapsed_ms,
                2,
            ),
            "guidance": "",
            "error": str(error),
        }

    elapsed_ms = (
        perf_counter() - start_time
    ) * 1000.0

    individual_checks = result[
        "individual_checks"
    ]

    if category == "good":
        correctly_classified = (
            result["passed"]
            and all(
                individual_checks.values()
            )
        )

    else:
        expected_defect_detected = not (
            individual_checks[
                expected_metric
            ]
        )

        correctly_classified = (
            expected_defect_detected
            and not result["passed"]
        )

    blur_result = result["blur"]
    brightness_result = result[
        "brightness"
    ]
    glare_result = result["glare"]
    roi_result = result["roi"]
    ridge_result = result["ridge"]

    return {
        "category": category,
        "filename": image_path.name,
        "expected_defect": (
            expected_metric or "none"
        ),
        "correctly_classified": bool(
            correctly_classified
        ),
        "final_passed": bool(
            result["passed"]
        ),

        # Composite scores
        "raw_score": result[
            "raw_composite_score"
        ],
        "final_score": result[
            "composite_score"
        ],

        # Blur details
        "blur_score": blur_result[
            "blur_score"
        ],
        "blur_threshold": blur_result[
            "threshold"
        ],
        "blur_strip_scores": "|".join(
            str(score)
            for score in blur_result[
                "strip_scores"
            ]
        ),
        "blur_passed": individual_checks[
            "blur"
        ],

        # Brightness details
        "finger_brightness": (
            brightness_result[
                "brightness"
            ]
        ),
        "global_brightness": (
            brightness_result[
                "global_brightness"
            ]
        ),
        "dark_pixel_fraction": (
            brightness_result[
                "dark_pixel_fraction"
            ]
        ),
        "bright_pixel_fraction": (
            brightness_result[
                "bright_pixel_fraction"
            ]
        ),
        "dark_threshold": (
            brightness_result[
                "dark_threshold"
            ]
        ),
        "bright_threshold": (
            brightness_result[
                "bright_threshold"
            ]
        ),
        "brightness_passed": (
            individual_checks[
                "brightness"
            ]
        ),

        # Glare details
        "glare_fraction": glare_result[
            "glare_fraction"
        ],
        "glare_fraction_threshold": (
            glare_result[
                "fraction_threshold"
            ]
        ),
        "roi_glare_fraction": (
            glare_result[
                "roi_glare_fraction"
            ]
        ),
        "roi_glare_fraction_threshold": (
            glare_result[
                "roi_fraction_threshold"
            ]
        ),
        "largest_hotspot_fraction": (
            glare_result[
                "largest_hotspot_fraction"
            ]
        ),
        "hotspot_fraction_threshold": (
            glare_result[
                "hotspot_fraction_threshold"
            ]
        ),
        "largest_hotspot_pixels": (
            glare_result[
                "largest_hotspot_pixels"
            ]
        ),
        "global_glare_detected": (
            glare_result[
                "global_glare_detected"
            ]
        ),
        "localized_glare_detected": (
            glare_result[
                "localized_glare_detected"
            ]
        ),
        "glare_passed": individual_checks[
            "glare"
        ],

        # ROI details
        "roi_fraction": roi_result[
            "roi_fraction"
        ],
        "roi_threshold": roi_result[
            "threshold"
        ],
        "roi_passed": individual_checks[
            "roi"
        ],

        # Ridge details
        "ridge_score": ridge_result[
            "ridge_score"
        ],
        "ridge_threshold": ridge_result[
            "threshold"
        ],
        "ridge_passed": individual_checks[
            "ridge"
        ],

        # Performance and guidance
        "processing_time_ms": round(
            elapsed_ms,
            2,
        ),
        "guidance": result[
            "guidance"
        ],
        "error": "",
    }

def evaluate_dataset() -> list[dict[str, Any]]:

    test_cases = discover_official_images()

    results = []

    for category, image_path in test_cases:
        result = evaluate_test_case(
            category,
            image_path,
        )

        results.append(result)

    return results

def save_evaluation_csv(
    results: list[dict[str, Any]],
    output_path: Path = OUTPUT_CSV,
) -> None:

    if not results:
        raise ValueError(
            "No evaluation results were provided."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

def print_evaluation_summary(
    results: list[dict[str, Any]],
) -> None:

    print("\nFingerprint quality evaluation")
    print("=" * 75)

    for result in results:
        status = (
            "PASS"
            if result["correctly_classified"]
            else "FAIL"
        )

        failed_checks = []

        for metric_name in METRIC_ORDER:
            value = result[
                f"{metric_name}_passed"
            ]

            if value is False:
                failed_checks.append(
                    metric_name
                )

        failed_text = (
            ", ".join(failed_checks)
            if failed_checks
            else "none"
        )

        print(
            f"[{status}] "
            f"{result['filename']:20s} "
            f"| expected: "
            f"{result['expected_defect']:10s} "
            f"| score: "
            f"{str(result['final_score']):6s} "
            f"| failed: {failed_text}"
        )

        if result["error"]:
            print(
                f"       Error: "
                f"{result['error']}"
            )

    print("\nCategory summary")
    print("-" * 45)

    for category in CATEGORIES:
        category_results = [
            result
            for result in results
            if result["category"] == category
        ]

        correct_count = sum(
            result["correctly_classified"]
            for result in category_results
        )

        print(
            f"{category:10s}: "
            f"{correct_count}/"
            f"{len(category_results)} correct"
        )

    total_correct = sum(
        result["correctly_classified"]
        for result in results
    )

    total_images = len(results)

    accuracy = (
        total_correct / total_images * 100.0
    )

    average_time = sum(
        float(result["processing_time_ms"])
        for result in results
    ) / total_images

    print("-" * 45)
    print(
        f"Overall: {total_correct}/"
        f"{total_images} correct"
    )
    print(
        f"Detection accuracy: "
        f"{accuracy:.2f}%"
    )
    print(
        f"Average processing time: "
        f"{average_time:.2f} ms"
    )

def test_official_quality_dataset() -> None:

    results = evaluate_dataset()

    failures = [
        (
            f"{result['filename']} "
            f"(expected {result['expected_defect']}, "
            f"guidance: {result['guidance']}, "
            f"error: {result['error']})"
        )
        for result in results
        if not result["correctly_classified"]
    ]

    assert not failures, (
        "Quality evaluation failures:\n"
        + "\n".join(failures)
    )

def main() -> None:
    """Run evaluation and save the results."""

    results = evaluate_dataset()

    print_evaluation_summary(results)

    save_evaluation_csv(
        results
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()