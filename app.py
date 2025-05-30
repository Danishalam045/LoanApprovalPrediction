# # app.py

# from flask import Flask, render_template, request
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the model
# model = joblib.load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from form
#         features = [
#             int(request.form['Gender']),
#             int(request.form['Married']),
#             int(request.form['Dependents']),
#             int(request.form['Education']),
#             int(request.form['Self_Employed']),
#             float(request.form['ApplicantIncome']),
#             float(request.form['CoapplicantIncome']),
#             float(request.form['LoanAmount']),
#             float(request.form['Loan_Amount_Term']),
#             float(request.form['Credit_History']),
#             int(request.form['Property_Area'])
#         ]

#         # Reshape and predict
#         input_data = np.array(features).reshape(1, -1)
#         prediction = model.predict(input_data)

#         # Decode prediction
#         result = 'Approved' if prediction[0] == 1 else 'Not Approved'
#         return render_template('index.html', prediction=result)
#     except Exception as e:
#         return f"Error: {e}"

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import joblib
import pandas as pd  # Add pandas import

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define feature names exactly as in training
        feature_names = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]

        # Extract features from form input
        features = [
            int(request.form['Gender']),
            int(request.form['Married']),
            int(request.form['Dependents']),
            int(request.form['Education']),
            int(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            int(request.form['Property_Area'])
        ]

        # Create a DataFrame with one row, passing column names
        input_df = pd.DataFrame([features], columns=feature_names)

        # Predict using DataFrame to avoid warning
        prediction = model.predict(input_df)

        result = 'Approved' if prediction[0] == 1 else 'Not Approved'
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
