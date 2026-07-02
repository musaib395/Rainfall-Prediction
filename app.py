from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load model + mapping
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    model, mapping = pickle.load(f)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    states = sorted(mapping['SUBDIVISION'].unique())

    if request.method == "POST":
        try:
            year = int(request.form["Year"])
            month = int(request.form["Month"])
            state = request.form["State"]

            #  Input Validation
            if year < 1901:
                return "❌ Year must be >= 1901"

            if month < 1 or month > 12:
                return "❌ Month must be between 1 and 12"

            #  Convert state to numeric code
            state_code = mapping[mapping['SUBDIVISION'] == state]['State_Code'].values[0]

            #  Predict rainfall
            prediction = model.predict([[state_code, year, month]])[0]
            prediction = round(prediction, 2)

            return render_template(
                "result.html",
                Year=year,
                Month=month,
                State=state,
                result=prediction
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", states=states)


# Run app
if __name__ == "__main__":
    app.run(debug=True)
