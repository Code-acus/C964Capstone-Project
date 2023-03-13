# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.

import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler


def predict_diabetes(input_df):
    # load the model
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'diabetes_model.joblib')
    model = load(model_path)

    # preprocess the input data
    scaler = StandardScaler()
    input_df = scaler.fit_transform(input_df)

    # predict the input data
    prediction = model.predict(input_df)
    if prediction == 0:
        return "You do not have diabetes"
    else:
        return "You have diabetes"


class App:
    def __init__(self, master):
        self.master = master

        # Create input fields and labels
        self.pregnancies_label = tk.Label(master, text="Pregnancies:")
        self.pregnancies_label.grid(row=0, column=0)
        self.pregnancies_entry = tk.Entry(master)
        self.pregnancies_entry.grid(row=0, column=1)

        self.glucose_label = tk.Label(master, text="Glucose:")
        self.glucose_label.grid(row=1, column=0)
        self.glucose_entry = tk.Entry(master)
        self.glucose_entry.grid(row=1, column=1)

        self.blood_pressure_label = tk.Label(master, text="Blood Pressure:")
        self.blood_pressure_label.grid(row=2, column=0)
        self.blood_pressure_entry = tk.Entry(master)
        self.blood_pressure_entry.grid(row=2, column=1)

        self.skin_thickness_label = tk.Label(master, text="Skin Thickness:")
        self.skin_thickness_label.grid(row=3, column=0)
        self.skin_thickness_entry = tk.Entry(master)
        self.skin_thickness_entry.grid(row=3, column=1)

        self.insulin_label = tk.Label(master, text="Insulin:")
        self.insulin_label.grid(row=4, column=0)
        self.insulin_entry = tk.Entry(master)
        self.insulin_entry.grid(row=4, column=1)

        self.bmi_label = tk.Label(master, text="BMI:")
        self.bmi_label.grid(row=5, column=0)
        self.bmi_entry = tk.Entry(master)
        self.bmi_entry.grid(row=5, column=1)

        self.diabetes_pedigree_function_label = tk.Label(master, text="Diabetes Pedigree Function:")
        self.diabetes_pedigree_function_label.grid(row=6, column=0)
        self.diabetes_pedigree_function_entry = tk.Entry(master)
        self.diabetes_pedigree_function_entry.grid(row=6, column=1)

        # Create the predict button
        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.grid(row=7, column=0, columnspan=2)

    def predict(self):
        try:
            input_data = {
                'Pregnancies': [int(self.pregnancies_entry.get())],
                'Glucose': [int(self.glucose_entry.get())],
                'BloodPressure': [int(self.blood_pressure_entry.get())],
                'SkinThickness': [int(self.skin_thickness_entry.get())],
                'Insulin': [int(self.insulin_entry.get())],
                'BMI': [float(self.bmi_entry.get())],
                'DiabetesPedigreeFunction': [
                    float(self.diabetes_pedigree_function_entry.get())]
            }
            input_df = pd.DataFrame(input_data)
            prediction = predict_diabetes(input_df)
            messagebox.showinfo("Prediction", prediction)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Diabetes Predictor")
    root.geometry("300x300")
    app = App(root)
    root.mainloop()

# Path: C964Capstone.py
# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.
