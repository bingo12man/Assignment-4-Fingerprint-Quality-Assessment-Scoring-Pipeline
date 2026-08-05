# Fingerprint Quality Assessment and Scoring Pipeline

A Python, OpenCV, and Streamlit application that checks whether a contactless fingerprint image is suitable for further biometric processing.

The pipeline evaluates five quality factors:

- Blur
- Brightness
- Glare
- Finger region-of-interest (ROI) completeness
- Fingerprint ridge clarity

It returns a quality score from 0 to 100, an accept/reject decision, individual metric results, processing time, and recapture guidance.

## Implemented Methods

| Stage | Method |
|---|---|
| Blur | Laplacian variance over three overlapping central image regions |
| Brightness | Mean grayscale intensity inside the detected finger ROI |
| Glare | Bright-pixel counting with global, ROI, and connected-hotspot checks |
| ROI | Otsu thresholding, morphology, and contour-based foreground selection |
| Ridge clarity | Multi-orientation Gabor filtering on the fingertip ridge core |
| Final decision | Weighted 0–100 score plus mandatory individual checks |

## Project Structure

```text
fingerprint-quality-assessment/
├── quality_assessment.py        # Metric functions and quality_gate()
├── quality_app.py               # Streamlit web interface
├── test_quality.py              # Evaluation on 20 test images
├── requirements.txt
├── README.md
├── report.pdf                   # Add before final submission
├── data/                        # Local fingerprint evaluation images
│   ├── good/
│   ├── blurry/
│   ├── dark/
│   └── glare/
├── outputs/
   └── quality_evaluation.csv   # Detailed results for all 20 images
   └──screenshots/              # Streamlit result screenshots
```

### Output Locations

- Evaluation CSV: `outputs/quality_evaluation.csv`
- Streamlit screenshots: `screenshots/`
- Test images: `data/good/`, `data/blurry/`, `data/dark/`, and `data/glare/`

Fingerprint images are biometric data. Keep the repository private or use sanitized images when sharing publicly.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bingo12man/Assignment-4-Fingerprint-Quality-Assessment-Scoring-Pipeline.git
cd Assignment-4-Fingerprint-Quality-Assessment-Scoring-Pipeline
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Use

### Run a single pipeline smoke test

```bash
python3 quality_assessment.py
```

This runs the quality gate on the default example image and prints the blur score, strip scores, composite score, final decision, and guidance.

### Evaluate all 20 assignment images

```bash
python3 test_quality.py
```

The script evaluates:

- 5 good images
- 5 blurry images
- 5 dark images
- 5 glare images

It prints category results and writes the detailed table to:

```text
outputs/quality_evaluation.csv
```

### Run automated tests

```bash
python3 -m pytest -q
```

Latest result:

```text
1 passed
```

The Pytest duration is the total time for the full test run, not the processing time for one image.

### Launch the Streamlit application

```bash
streamlit run quality_app.py
```

Then open the local URL shown in the terminal, upload a PNG/JPG/JPEG image, and review:

- Uploaded-image preview
- Final quality score and raw weighted score
- Accept/reject decision
- Blur, brightness, glare, ROI, and ridge results
- Processing time
- Capture guidance
- Adjustable thresholds in the sidebar

## Calibrated Thresholds

| Check | Default value |
|---|---:|
| Main blur threshold | 300 |
| Severe blur threshold | 70 |
| Blur strip spread threshold | 80 |
| Dark brightness threshold | 80 |
| Bright brightness threshold | 210 |
| Glare pixel threshold | 240 |
| Global glare fraction | 0.009 |
| ROI glare fraction | 0.007 |
| Glare hotspot fraction | 0.0045 |
| Minimum ROI fraction | 0.15 |
| Ridge clarity threshold | 1.0 |
| Composite acceptance threshold | 60 |

An image is accepted only when its final score is at least 60 and every mandatory individual check passes.

## Evaluation Result

Latest evaluation:

```text
Good:    5/5 correct
Blurry:  5/5 correct
Dark:    5/5 correct
Glare:   5/5 correct

Overall: 20/20 correct
Detection accuracy: 100.00%
Average processing time: 142.90 ms per image
```

The total processing requirement of less than 300 ms per image is satisfied on the current evaluation machine.

The reported 100% accuracy applies only to the current 20-image assignment dataset, which was also used during threshold calibration. It should not be interpreted as general accuracy across all cameras, skin tones, backgrounds, and lighting conditions.

## Capture Guidance Examples

The application returns messages such as:

- `Good capture — ready for processing`
- `Image is too blurry — hold your hand steady and retry.`
- `Image is too dark — increase the lighting and retry.`
- `Glare detected — change the light angle and retry.`
- `Finger is too small — move it closer to the camera.`
- `Fingerprint ridges are unclear — adjust focus and distance.`

## Limitations

- Thresholds were calibrated using a small dataset.
- Complex backgrounds can affect foreground segmentation.
- Results may vary with camera quality, compression, distance, and lighting.
- A separate unseen test set is needed for a stronger generalization estimate.
- Naturally worn fingerprints may require adaptive thresholds or an alternative verification path.
