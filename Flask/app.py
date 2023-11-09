import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained XGBoost model
with open("xgb_model.pkl", "rb") as model_file:
    xgboost_model = pickle.load(model_file)

# Load the pickled StandardScaler object
with open("standard_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        features = {
            "Warehouse_block": request.form["Warehouse_block"],
            "Mode_of_Shipment": request.form["Mode_of_Shipment"],
            "Customer_care_calls": int(request.form["Customer_care_calls"]),
            "Customer_rating": int(request.form["Customer_rating"]),
            "Cost_of_the_Product": int(request.form["Cost_of_the_Product"]),
            "Prior_purchases": int(request.form["Prior_purchases"]),
            "Product_importance": request.form["Product_importance"],
            "Gender": request.form["Gender"],
            "Discount_offered": int(request.form["Discount_offered"]),
            "Weight_in_gms": int(request.form["Weight_in_gms"]),
        }

        # Create a DataFrame from the user input
        input_data = pd.DataFrame([features])

        # Manually label the categorical columns
        input_data['Warehouse_block'] = input_data['Warehouse_block'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4})
        input_data['Mode_of_Shipment'] = input_data['Mode_of_Shipment'].map({'Ship': 0, 'Flight': 1, 'Road': 2})
        input_data['Product_importance'] = input_data['Product_importance'].map({'low': 0, 'medium': 1, 'high': 2})
        input_data['Gender'] = input_data['Gender'].map({'F': 0, 'M': 1})


        # Standardize numerical features using the loaded StandardScaler
        input_data_scaled = scaler.transform(input_data)

        # Use the XGBoost model for prediction
        prediction = xgboost_model.predict(input_data_scaled)

        # Display the prediction on the web page
        result = "On-Time Delivery" if prediction[0] == 1 else "Late Delivery"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
