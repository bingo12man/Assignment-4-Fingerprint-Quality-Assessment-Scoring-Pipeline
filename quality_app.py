from pathlib import Path
from time import perf_counter
import os
import tempfile

import streamlit as st

from quality_assessment import (
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_SEVERE_BLUR_THRESHOLD,
    DEFAULT_BLUR_SPREAD_THRESHOLD,
    DEFAULT_GLARE_PIXEL_THRESHOLD,
    DEFAULT_GLARE_FRACTION_THRESHOLD,
    DEFAULT_GLARE_SATURATION_THRESHOLD,
    DEFAULT_ROI_GLARE_FRACTION_THRESHOLD,
    DEFAULT_GLARE_HOTSPOT_FRACTION_THRESHOLD,
    DEFAULT_ROI_FRACTION_THRESHOLD,
    DEFAULT_RIDGE_CLARITY_THRESHOLD,
    DEFAULT_COMPOSITE_THRESHOLD,
    quality_gate,
)


st.set_page_config(
    page_title="Fingerprint Quality Assessment",
    page_icon="🔍",
    layout="wide",
)


def render_metric(
    title: str,
    passed: bool,
    value: str,
    threshold: str,
) -> None:
    """Display one quality-metric result."""

    st.subheader(title)

    if passed:
        st.success("PASS")
    else:
        st.error("FAIL")

    st.metric(
        label="Measured value",
        value=value,
    )

    st.caption(
        threshold
    )


def save_uploaded_image(
    uploaded_file,
) -> str:
    """Save a Streamlit upload to a temporary local file."""

    file_suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if file_suffix not in {
        ".png",
        ".jpg",
        ".jpeg",
    }:
        raise ValueError(
            "Only PNG, JPG and JPEG images are supported."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_suffix,
    ) as temporary_file:
        temporary_file.write(
            uploaded_file.getbuffer()
        )

        return temporary_file.name


def main() -> None:
    """Run the Streamlit fingerprint-quality application."""

    st.title(
        "Fingerprint Quality Assessment"
    )

    st.write(
        "Upload a fingerprint image to evaluate blur, "
        "brightness, glare, ROI completeness and ridge clarity."
    )

    with st.sidebar:
        st.header(
            "Quality thresholds"
        )

        st.caption(
            "The default values were calibrated using "
            "the 20-image evaluation dataset."
        )

        st.subheader(
            "Blur"
        )

        blur_threshold = st.slider(
            "Main blur threshold",
            min_value=100.0,
            max_value=600.0,
            value=float(
                DEFAULT_BLUR_THRESHOLD
            ),
            step=10.0,
        )

        severe_maximum = min(
            250.0,
            blur_threshold - 1.0,
        )

        severe_default = min(
            float(
                DEFAULT_SEVERE_BLUR_THRESHOLD
            ),
            severe_maximum,
        )

        severe_blur_threshold = st.slider(
            "Severe blur threshold",
            min_value=10.0,
            max_value=float(
                severe_maximum
            ),
            value=float(
                severe_default
            ),
            step=5.0,
        )

        blur_spread_threshold = st.slider(
            "Blur strip spread threshold",
            min_value=0.0,
            max_value=300.0,
            value=float(
                DEFAULT_BLUR_SPREAD_THRESHOLD
            ),
            step=5.0,
        )

        st.subheader(
            "Brightness"
        )

        dark_threshold = st.slider(
            "Dark threshold",
            min_value=0.0,
            max_value=149.0,
            value=80.0,
            step=1.0,
        )

        bright_threshold = st.slider(
            "Bright threshold",
            min_value=150.0,
            max_value=255.0,
            value=210.0,
            step=1.0,
        )

        st.subheader(
            "Glare"
        )

        glare_pixel_threshold = st.slider(
            "Bright-pixel threshold",
            min_value=200,
            max_value=255,
            value=int(
                DEFAULT_GLARE_PIXEL_THRESHOLD
            ),
            step=1,
        )

        glare_saturation_threshold = st.slider(
            "Maximum glare saturation",
            min_value=0,
            max_value=255,
            value=int(
                DEFAULT_GLARE_SATURATION_THRESHOLD
            ),
            step=1,
        )

        glare_fraction_threshold = st.slider(
            "Global glare fraction",
            min_value=0.0,
            max_value=0.050,
            value=float(
                DEFAULT_GLARE_FRACTION_THRESHOLD
            ),
            step=0.001,
            format="%.3f",
        )

        roi_glare_fraction_threshold = st.slider(
            "ROI glare fraction",
            min_value=0.0,
            max_value=0.050,
            value=float(
                DEFAULT_ROI_GLARE_FRACTION_THRESHOLD
            ),
            step=0.001,
            format="%.3f",
        )

        glare_hotspot_fraction_threshold = st.slider(
            "Hotspot fraction",
            min_value=0.0,
            max_value=0.020,
            value=float(
                DEFAULT_GLARE_HOTSPOT_FRACTION_THRESHOLD
            ),
            step=0.0005,
            format="%.4f",
        )

        st.subheader(
            "ROI and ridge clarity"
        )

        roi_fraction_threshold = st.slider(
            "Minimum ROI fraction",
            min_value=0.05,
            max_value=0.50,
            value=float(
                DEFAULT_ROI_FRACTION_THRESHOLD
            ),
            step=0.01,
            format="%.2f",
        )

        ridge_threshold = st.slider(
            "Ridge clarity threshold",
            min_value=0.10,
            max_value=5.00,
            value=float(
                DEFAULT_RIDGE_CLARITY_THRESHOLD
            ),
            step=0.10,
            format="%.2f",
        )

        composite_threshold = st.slider(
            "Composite acceptance threshold",
            min_value=0.0,
            max_value=100.0,
            value=float(
                DEFAULT_COMPOSITE_THRESHOLD
            ),
            step=1.0,
        )

    uploaded_file = st.file_uploader(
        "Upload fingerprint image",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
    )

    if uploaded_file is None:
        st.info(
            "Upload an image to begin the quality assessment."
        )

        return

    image_column, result_column = st.columns(
        [1, 1]
    )

    with image_column:
        st.subheader(
            "Uploaded image"
        )

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    temporary_path = None

    try:
        temporary_path = save_uploaded_image(
            uploaded_file
        )

        start_time = perf_counter()

        result = quality_gate(
            temporary_path,
            blur_threshold=blur_threshold,
            severe_blur_threshold=(
                severe_blur_threshold
            ),
            blur_spread_threshold=(
                blur_spread_threshold
            ),
            dark_threshold=dark_threshold,
            bright_threshold=bright_threshold,
            glare_pixel_threshold=(
                glare_pixel_threshold
            ),
            glare_fraction_threshold=(
                glare_fraction_threshold
            ),
            glare_saturation_threshold=(
                glare_saturation_threshold
            ),
            roi_glare_fraction_threshold=(
                roi_glare_fraction_threshold
            ),
            glare_hotspot_fraction_threshold=(
                glare_hotspot_fraction_threshold
            ),
            roi_fraction_threshold=(
                roi_fraction_threshold
            ),
            ridge_threshold=ridge_threshold,
            composite_threshold=(
                composite_threshold
            ),
        )

        processing_time_ms = (
            perf_counter() - start_time
        ) * 1000.0

    except (
        ValueError,
        OSError,
    ) as error:
        st.error(
            f"Quality assessment failed: {error}"
        )

        return

    finally:
        if (
            temporary_path is not None
            and os.path.exists(
                temporary_path
            )
        ):
            os.remove(
                temporary_path
            )

    with result_column:
        st.subheader(
            "Final assessment"
        )

        if result["passed"]:
            st.success(
                "ACCEPTABLE — fingerprint image passed "
                "the quality gate."
            )
        else:
            st.error(
                "REJECTED — fingerprint image requires "
                "another capture."
            )

        score_column, raw_score_column = st.columns(
            2
        )

        with score_column:
            st.metric(
                "Final quality score",
                f"{result['composite_score']:.2f}/100",
            )

        with raw_score_column:
            st.metric(
                "Raw weighted score",
                f"{result['raw_composite_score']:.2f}/100",
            )

        progress_value = int(
            round(
                result["composite_score"]
            )
        )

        progress_value = max(
            0,
            min(
                100,
                progress_value,
            ),
        )

        st.progress(
            progress_value
        )

        st.caption(
            f"Acceptance threshold: "
            f"{result['composite_threshold']:.2f}"
        )

        st.metric(
            "Processing time",
            f"{processing_time_ms:.2f} ms",
        )

        if processing_time_ms <= 300.0:
            st.success(
                "Processing time is within the "
                "300 ms performance budget."
            )
        else:
            st.warning(
                "Processing time exceeded the "
                "300 ms performance budget."
            )

    st.divider()

    st.header(
        "Individual quality metrics"
    )

    metric_columns = st.columns(
        5
    )

    blur_result = result["blur"]
    brightness_result = result[
        "brightness"
    ]
    glare_result = result["glare"]
    roi_result = result["roi"]
    ridge_result = result["ridge"]

    with metric_columns[0]:
        render_metric(
            title="Blur",
            passed=result[
                "individual_checks"
            ]["blur"],
            value=f"{blur_result['blur_score']:.2f}",
            threshold=(
                f"Main ≥ {blur_result['threshold']:.0f}; "
                f"partial-region rule uses severe ≥ "
                f"{blur_result['severe_threshold']:.0f} "
                f"and spread > "
                f"{blur_result['spread_threshold']:.0f}"
            ),
        )

    with metric_columns[1]:
        render_metric(
            title="Brightness",
            passed=result[
                "individual_checks"
            ]["brightness"],
            value=(
                f"{brightness_result['brightness']:.2f}"
            ),
            threshold=(
                f"Allowed range: "
                f"{brightness_result['dark_threshold']:.0f}"
                f"–"
                f"{brightness_result['bright_threshold']:.0f}"
            ),
        )

    with metric_columns[2]:
        render_metric(
            title="Glare",
            passed=result[
                "individual_checks"
            ]["glare"],
            value=(
                f"{glare_result['glare_percentage']:.2f}%"
            ),
            threshold=(
                f"Global: "
                f"{glare_result['fraction_threshold'] * 100:.2f}% | "
                f"ROI: "
                f"{glare_result['roi_fraction_threshold'] * 100:.2f}% | "
                f"hotspot: "
                f"{glare_result['hotspot_fraction_threshold'] * 100:.2f}%"
            ),
        )

    with metric_columns[3]:
        render_metric(
            title="ROI",
            passed=result[
                "individual_checks"
            ]["roi"],
            value=(
                f"{roi_result['roi_percentage']:.2f}%"
            ),
            threshold=(
                f"Minimum: "
                f"{roi_result['threshold'] * 100:.2f}%"
            ),
        )

    with metric_columns[4]:
        render_metric(
            title="Ridge clarity",
            passed=result[
                "individual_checks"
            ]["ridge"],
            value=(
                f"{ridge_result['ridge_score']:.2f}"
            ),
            threshold=(
                f"Minimum: "
                f"{ridge_result['threshold']:.2f}"
            ),
        )

    st.divider()

    details_column, guidance_column = st.columns(
        2
    )

    with details_column:
        st.subheader(
            "Detailed measurements"
        )

        st.write(
            "**Blur strip scores:** "
            + ", ".join(
                f"{score:.2f}"
                for score in blur_result[
                    "strip_scores"
                ]
            )
        )

        st.write(
            "**Blur strip spread:** "
            f"{blur_result['strip_score_spread']:.2f}"
        )

        st.write(
            "**Blur decision method:** "
            + (
                "Main median threshold"
                if blur_result["median_score_passed"]
                else "Partial-region rule"
                if blur_result["partial_finger_passed"]
                else "Failed both blur rules"
            )
        )
        st.write(
            "**Finger brightness:** "
            f"{brightness_result['brightness']:.2f}"
        )

        st.write(
            "**Global brightness:** "
            f"{brightness_result['global_brightness']:.2f}"
        )

        st.write(
            "**ROI glare:** "
            f"{glare_result['roi_glare_percentage']:.2f}%"
        )

        st.write(
            "**Largest glare hotspot:** "
            f"{glare_result['largest_hotspot_fraction'] * 100:.3f}%"
        )

        st.write(
            "**ROI coverage:** "
            f"{roi_result['roi_percentage']:.2f}%"
        )

        st.write(
            "**Ridge score:** "
            f"{ridge_result['ridge_score']:.4f}"
        )

    with guidance_column:
        st.subheader(
            "Capture guidance"
        )

        if result["passed"]:
            st.success(
                result["guidance"]
            )
        else:
            st.warning(
                result["guidance"]
            )

            for issue in result["issues"]:
                st.write(
                    f"• {issue}"
                )

    with st.expander(
        "Normalized scores and contributions"
    ):
        normalized_metrics = result[
            "normalized_metrics"
        ]

        contributions = result[
            "weighted_contributions"
        ]

        for metric_name in (
            "blur",
            "brightness",
            "glare",
            "roi",
            "ridge",
        ):
            st.write(
                f"**{metric_name.title()}** — "
                f"normalized: "
                f"{normalized_metrics[metric_name]:.4f}, "
                f"weighted contribution: "
                f"{contributions[metric_name]:.2f}"
            )


if __name__ == "__main__":
    main()