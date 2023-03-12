# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.

import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split


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
                'DiabetesPedigreeFunction': [float(self.diabetes_pedigree_function_entry.get())]
            }
            input_df = pd.DataFrame(input_data)
            prediction = self.predict_diabetes(input_df)
            messagebox.showinfo("Prediction", prediction)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values")

    def predict_diabetes(self, input_df):
        # Get the absolute path of the csv file
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "diabetes_data.csv"))
        # Read the data from the csv file
        df = pd.read_csv(csv_path)
        # Split the data into features and labels
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # Create the model
        model = svm.SVC(kernel='linear')
        # Train the model
        model.fit(X_train, y_train)

        # Predict the outcome
        prediction = model.predict(input_df)
        if prediction == 1:
            return "Diabetes"
        else:
            return "No Diabetes"

root = tk.Tk()
root.title("Diabetes Predictor")
app = App(root)
root.mainloop()

# Path: C964Capstone.py
# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.
