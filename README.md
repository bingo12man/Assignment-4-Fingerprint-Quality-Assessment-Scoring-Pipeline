# Fingerprint Quality Assessment Pipeline

A Python and OpenCV-based system that checks whether a fingerprint image is suitable for biometric processing.

The pipeline evaluates:

- Blur
- Brightness
- Glare
- Finger ROI completeness
- Ridge clarity

It produces a quality score from 0 to 100, an accept/reject decision, and recapture guidance.

## Features

- Laplacian variance blur detection
- Finger-region brightness analysis
- Global and localized glare detection
- Finger ROI segmentation
- Gabor-filter ridge clarity measurement
- Weighted composite quality score
- Streamlit interface with adjustable thresholds
- Automated evaluation using 20 test images
- Processing time below 300 ms per image

## Project Structure

```text
fingerprint-quality-assessment/
├── quality_assessment.py
├── quality_app.py
├── test_quality.py
├── requirements.txt
├── README.md
├── data/
│   ├── good/
│   ├── blurry/
│   ├── dark/
│   └── glare/
├── outputs/
└── screenshots/