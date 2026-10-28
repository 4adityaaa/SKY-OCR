# SkyOCR

SkyOCR is a Flask web app for extracting text from images with Tesseract. It can enhance image contrast before OCR and optionally compare the result with supplied ground-truth text.

## Features

- Upload or drag and drop PNG, JPG, BMP, TIFF, and WEBP images.
- Preview the selected image before processing.
- Choose a Tesseract page segmentation mode.
- Optionally enhance contrast with CLAHE preprocessing.
- View extracted text, Tesseract confidence, and detected word count.
- Compare the extracted text with optional ground truth using character and word accuracy metrics.
- Switch between day and night themes.

## Requirements

- Python 3.10 or later
- Tesseract OCR installed and available on your `PATH`
- Python packages: Flask, opencv-python, numpy, and pytesseract

On Windows, install Tesseract OCR and make sure `tesseract.exe` is on your `PATH`. The app also checks the default installation path `C:\Program Files\Tesseract-OCR\tesseract.exe`.

## Setup and run

From the directory containing this README, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install Flask opencv-python numpy pytesseract
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The Flask server is configured for local development; do not expose its debug server to an untrusted network.

If PowerShell prevents virtual environment activation, you can install the packages without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install Flask opencv-python numpy pytesseract
.\.venv\Scripts\python.exe app.py
```

## Using the app

1. Select an image or drag it onto the upload area. Images must be smaller than 16 MB.
2. Choose a page segmentation mode appropriate for the layout: automatic, column, text block, single line, or sparse text.
3. Leave contrast enhancement enabled for the default preprocessing, or turn it off.
4. Optionally enter the correct text to calculate character and word accuracy.
5. Select **Extract Text**. Copy the resulting text with **Copy text**.

Without ground-truth text, the app still reports OCR confidence and word count, but accuracy metrics are unavailable.