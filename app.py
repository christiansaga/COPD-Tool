from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model
model = joblib.load("model/logistic_model.pkl")  # Adjust the path to your saved model

# Define the route for the homepage
@app.route("/")
def home():
    return render_template("index.html")  # Ensure you have an index.html in the templates folder

# Define the route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debug: Log the received form data
        print("Form Data Received:", request.form)

        # Parse input data from form
        data = [
            float(request.form["Shortness_of_Breath"]),
            float(request.form["Cough_Intensity"]),
            float(request.form["Chest_Tightness"]),
            float(request.form["Wheezing"]),
            float(request.form["Fatigue"]),
            float(request.form["Age"]),
            float(request.form["Respiratory_Rate"]),
            int(request.form["Smoking_History"]),
            int(request.form["Comorbidities"]),
            int(request.form["Mucus_Color"]),
            float(request.form["Mucus_Amount"]),
            int(request.form["Fever_Last_2_Weeks"]),
        ]

        # Debug: Log parsed data
        print("Parsed Data:", data)

        # Reshape data for the model
        data = np.array(data).reshape(1, -1)

        # Debug: Log reshaped data
        print("Input Shape for Model:", data.shape)
        print("Input Data for Model:", data)

        # Debug: Predict probabilities for both classes
        probabilities = model.predict_proba(data)
        print("Probabilities for Both Classes:", probabilities)

        # Extract probability for admission (class 1)
        probability = probabilities[0][1]
        print("Final Probability for Admission:", probability)

        # Generate prediction based on probability
        if probability > 0.91   :
            prediction = "Should be considered for hospital admission"
        else:
            prediction = "Continue to monitor symptoms."

        # Map probability to severity levels
        if probability <= 0.4:
            severity = "Mild"
        elif 0.4 < probability <= 0.9:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Debug: Log severity level
        print("Severity Level:", severity)

        return render_template(
                "index.html",
                prediction_text=f"Prediction: {prediction}",
                severity_text=f"Severity Level: {severity}",
                #probability_text=f"Probability: {probability:.2%}"
            )
    except Exception as e:
        # Debug: Log the exception
        print("Error occurred:", str(e))
        return render_template("index.html", error_text=f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)  # Bind to 0.0.0.0 for Heroku