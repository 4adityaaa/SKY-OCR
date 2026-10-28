"""
app.py - Flask server for the OCR web UI.
Run:  python app.py   ->  open http://127.0.0.1:5000
"""
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from ocr_engine import run_ocr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
SUPPORTED_PSM_MODES = {3, 4, 6, 7, 11}


@app.route("/")
def index():
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify(error="Image is too large. Please upload an image smaller than 16 MB."), 413


@app.route("/ocr", methods=["POST"])
def ocr():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="No image uploaded."), 400
    contents = file.read()
    if not contents:
        return jsonify(error="The uploaded file is empty."), 400

    data = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not read that file as an image."), 400

    ground_truth = request.form.get("ground_truth", "").strip()
    try:
        psm = int(request.form.get("psm", 6))
    except ValueError:
        return jsonify(error="Invalid page segmentation mode."), 400
    if psm not in SUPPORTED_PSM_MODES:
        return jsonify(error="Unsupported page segmentation mode."), 400
    pre = request.form.get("preprocess", "true") == "true"

    try:
        result, _ = run_ocr(img, ground_truth=ground_truth, psm=psm, preprocess_image=pre)
    except (RuntimeError, OSError, cv2.error) as error:
        app.logger.exception("OCR engine failed")
        return jsonify(error=f"OCR failed. Check that Tesseract is installed and working. Details: {error}"), 500

    has_gt = bool(ground_truth)
    return jsonify(
        text=result.text,
        confidence=round(result.confidence, 1),
        word_count=result.word_count,
        has_ground_truth=has_gt,
        char_accuracy=round(result.char_accuracy, 1) if has_gt else None,
        word_accuracy=round(result.word_accuracy, 1) if has_gt else None,
        char_distance=result.char_distance if has_gt else None,
    )


if __name__ == "__main__":
    app.run(debug=True)
