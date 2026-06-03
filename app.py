from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model once
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        filename = secure_filename(file.filename)

        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Run YOLO prediction
        results = model(upload_path)

        # Get plotted image
        plotted_img = results[0].plot()

        output_filename = "result_" + filename
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        cv2.imwrite(output_path, plotted_img)

        output_image = output_filename

    return render_template("index.html", output_image=output_image)

if __name__ == "__main__":
    app.run(debug=True)
    