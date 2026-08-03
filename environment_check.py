import cv2
import numpy as np
import pandas as pd
import streamlit
import pytest

def main() -> None:
    """Verify package versions."""

    print("Environment setup successful")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Streamlit version: {streamlit.__version__}")
    print(f"Pytest version: {pytest.__version__}")


if __name__ == "__main__":
    main()
    
