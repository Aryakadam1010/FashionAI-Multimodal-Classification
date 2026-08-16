from flask import Flask, render_template, request
from predict import predict
import os
import uuid


app = Flask(__name__)

# ==========================================
# Configuration
# ==========================================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# Home
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        image = request.files.get("image")

        product_name = request.form.get(
            "product_name"
        )

        # ------------------------------
        # Validation
        # ------------------------------

        if image is None or image.filename == "":
            error = "Please upload a product image."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error=error
            )

        if not product_name or not product_name.strip():
            error = "Please enter the product name."

            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                image_url=image_url,
                error=error
            )

        # ------------------------------
        # Save Image
        # ------------------------------

        extension = os.path.splitext(
            image.filename
        )[1]

        filename = (
            str(uuid.uuid4()) + extension
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        image.save(image_path)

        # URL used by browser
        image_url = "/static/uploads/" + filename

        # ------------------------------
        # Prediction
        # ------------------------------

        try:

            prediction, confidence = predict(
                image_path,
                product_name
            )

            confidence = round(
                confidence * 100,
                2
            )

        except Exception as e:

            error = (
                "Prediction failed: "
                + str(e)
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )