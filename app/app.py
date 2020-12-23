from flask import Flask, render_template, request, jsonify
from torch_utils import transform, predict
import torch
from PIL import Image
import numpy as np
import io

app = Flask(__name__)


def check_if_allowed(f):
    filename = f.filename
    ext = filename.split(".")[1]
    if ext not in ["jpg", "jpeg", "png"]:
        return False
    else:
        return True


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", resp=None)
    elif request.method == "POST":
        f = request.files["file"]
        if f is None or f.filename == "":
            return jsonify({"error": "no file provided"})
        if not check_if_allowed(f):
            return jsonify({"error": "filetype is not allowed"})

        # Inference routine
        image_bytes = f.read()
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        transformed = transform(image)
        transformed = np.transpose(transformed, (2, 0, 1)).astype(np.float32)

        tensor_img = torch.tensor(transformed, dtype=torch.float)
        prediction = predict(tensor_img)

        return render_template("index.html", resp={"data": prediction.item()})


if __name__ == "__main__":
    app.run(debug=True)
