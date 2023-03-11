# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.

import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

class App:
    def __init__(self, master):
        self.master = master

    def predict(self):
        try:
            input_data = {
                'Pregnancies': [int(pregnancies_entry.get())],
                'Glucose': [int(glucose_entry.get())],
                'BloodPressure': [int(blood_pressure_entry.get())],
                'SkinThickness': [int(skin_thickness_entry.get())],
                'Insulin': [int(insulin_entry.get())],
                'BMI': [float(bmi_entry.get())],
                'DiabetesPedigreeFunction': [float(diabetes_pedigree_function_entry.get())]
            }
            input_df = pd.DataFrame(input_data)
            prediction = App.predict_diabetes(input_df)
            messagebox.showinfo("Prediction", "The patient is " + prediction)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values")

    def predict_diabetes(input_df):
        # Load the dataset
        df = pd.read_csv('diabetes.csv')

        # Split the data into training and testing sets
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train the model
        model = svm.SVC(kernel='linear')
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Return the prediction
        return model.predict(input_df)

# Create the GUI
root = tk.Tk()
root.title("Diabetes Predictor")

# Create input fields and labels
pregnancies_label = tk.Label(root, text="Pregnancies:")
pregnancies_label.grid(row=0, column=0)
pregnancies_entry = tk.Entry(root)
pregnancies_entry.grid(row=0, column=1)

glucose_label = tk.Label(root, text="Glucose:")
glucose_label.grid(row=1, column=0)
glucose_entry = tk.Entry(root)
glucose_entry.grid(row=1, column=1)

blood_pressure_label = tk.Label(root, text="Blood Pressure:")
blood_pressure_label.grid(row=2, column=0)
blood_pressure_entry = tk.Entry(root)
blood_pressure_entry.grid(row=2, column=1)

skin_thickness_label = tk.Label(root, text="Skin Thickness:")
skin_thickness_label.grid(row=3, column=0)
skin_thickness_entry = tk.Entry(root)
skin_thickness_entry.grid(row=3, column=1)

insulin_label = tk.Label(root, text="Insulin:")
insulin_label.grid(row=4, column=0)
insulin_entry = tk.Entry(root)
insulin_entry.grid(row=4, column=1)

bmi_label = tk.Label(root, text="BMI:")
bmi_label.grid(row=5, column=0)
bmi_entry = tk.Entry(root)
bmi_entry.grid(row=5, column=1)

diabetes_pedigree_function_label = tk.Label(root, text="Diabetes Pedigree Function:")
diabetes_pedigree_function_label.grid(row=6, column=0)
diabetes_pedigree_function_entry = tk.Entry(root)
diabetes_pedigree_function_entry.grid(row=6, column=1)

# Create the predict button
predict_button = tk.Button(root, text="Predict", command=App.predict)
predict_button.grid(row=7, column=0, columnspan=2)

# Run the app
app = App(root)
root.mainloop()

# Path: C964Capstone.py
# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.
