"""
ocr_engine.py
-------------
Core OCR + accuracy logic. Kept separate from UI for testability.
"""

import cv2
import numpy as np
import pytesseract
import os
import platform
import shutil
from dataclasses import dataclass


# ------------------------------------------------------------
# Tesseract path (auto-detect by OS)
# ------------------------------------------------------------
def _configure_tesseract() -> None:
    """Set Tesseract binary path based on OS."""
    executable = shutil.which("tesseract")
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
        return

    if platform.system() == "Windows":
        windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(windows_path):
            pytesseract.pytesseract.tesseract_cmd = windows_path

_configure_tesseract()


# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------
@dataclass
class OCRResult:
    text: str
    char_accuracy: float
    word_accuracy: float
    char_distance: int
    confidence: float
    word_count: int


# ------------------------------------------------------------
# Levenshtein distance (pure Python — no external dep needed)
# ------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev = curr
    return prev[-1]


# ------------------------------------------------------------
# Accuracy metrics
# ------------------------------------------------------------
def character_accuracy(ground_truth: str, predicted: str) -> float:
    gt = ground_truth.strip()
    pred = predicted.strip()
    if not gt:
        return 0.0
    dist = _levenshtein(gt, pred)
    return max(0.0, (1 - dist / len(gt)) * 100)


def word_accuracy(ground_truth: str, predicted: str) -> float:
    gt_words = ground_truth.strip().split()
    pred_words = predicted.strip().split()
    if not gt_words:
        return 0.0

    previous = list(range(len(pred_words) + 1))
    for i, gt_word in enumerate(gt_words, 1):
        current = [i] + [0] * len(pred_words)
        for j, pred_word in enumerate(pred_words, 1):
            cost = 0 if gt_word == pred_word else 1
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
        previous = current
    return max(0.0, (1 - previous[-1] / len(gt_words)) * 100)


# ------------------------------------------------------------
# Image preprocessing (lightweight — keeps things fast)
# ------------------------------------------------------------
def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Upscale small images and improve grayscale contrast for OCR."""
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side < 1600:
        scale = min(2.0, 1600 / longest_side)
        image_bgr = cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def run_ocr(
    image_bgr: np.ndarray,
    ground_truth: str = "",
    psm: int = 6,
    preprocess_image: bool = True,
) -> tuple[OCRResult, np.ndarray]:
    """
    Run OCR on an image and return results + the image actually used.

    Args:
        image_bgr: OpenCV image (BGR)
        ground_truth: user-provided correct text
        psm: Tesseract page segmentation mode
        preprocess_image: whether to apply CLAHE contrast boost

    Returns:
        (OCRResult, processed_image)
    """
    processed = preprocess(image_bgr) if preprocess_image else cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2GRAY
    )

    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"

    data = pytesseract.image_to_data(
        processed, config=config, output_type=pytesseract.Output.DICT
    )
    lines: dict[tuple[int, int, int], list[str]] = {}
    confidences = []
    for text, confidence, block, paragraph, line in zip(
        data["text"],
        data["conf"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
    ):
        word = text.strip()
        if not word:
            continue
        key = (block, paragraph, line)
        lines.setdefault(key, []).append(word)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        if confidence_value >= 0:
            confidences.append(confidence_value)

    text = "\n".join(" ".join(words) for words in lines.values()).strip()
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    word_count = sum(len(words) for words in lines.values())

    # Accuracy metrics
    char_acc = character_accuracy(ground_truth, text)
    word_acc = word_accuracy(ground_truth, text)
    char_dist = _levenshtein(ground_truth.strip(), text)

    result = OCRResult(
        text=text,
        char_accuracy=char_acc,
        word_accuracy=word_acc,
        char_distance=char_dist,
        confidence=avg_conf,
        word_count=word_count,
    )
    return result, processed
