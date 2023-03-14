# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.

# import os
# import tkinter as tk
# from tkinter import messagebox
# import pandas as pd
# from joblib import load
# from sklearn.preprocessing import StandardScaler
#
#
# def predict_diabetes(input_df):
#     # load the model
#     model_path = os.path.join(os.path.dirname(__file__), 'model', 'diabetes_model.joblib')
#     model = load(model_path)
#
#     # preprocess the input data
#     scaler = StandardScaler()
#     input_df = scaler.fit_transform(input_df)
#
#     # predict the input data
#     prediction = model.predict(input_df)
#     if prediction == 0:
#         return "You do not have diabetes"
#     else:
#         return "You have diabetes"
#
#
# class App:
#     def __init__(self, master):
#         self.master = master
#
#         # Create input fields and labels
#         self.pregnancies_label = tk.Label(master, text="Pregnancies:")
#         self.pregnancies_label.grid(row=0, column=0)
#         self.pregnancies_entry = tk.Entry(master)
#         self.pregnancies_entry.grid(row=0, column=1)
#
#         self.glucose_label = tk.Label(master, text="Glucose:")
#         self.glucose_label.grid(row=1, column=0)
#         self.glucose_entry = tk.Entry(master)
#         self.glucose_entry.grid(row=1, column=1)
#
#         self.blood_pressure_label = tk.Label(master, text="Blood Pressure:")
#         self.blood_pressure_label.grid(row=2, column=0)
#         self.blood_pressure_entry = tk.Entry(master)
#         self.blood_pressure_entry.grid(row=2, column=1)
#
#         self.skin_thickness_label = tk.Label(master, text="Skin Thickness:")
#         self.skin_thickness_label.grid(row=3, column=0)
#         self.skin_thickness_entry = tk.Entry(master)
#         self.skin_thickness_entry.grid(row=3, column=1)
#
#         self.insulin_label = tk.Label(master, text="Insulin:")
#         self.insulin_label.grid(row=4, column=0)
#         self.insulin_entry = tk.Entry(master)
#         self.insulin_entry.grid(row=4, column=1)
#
#         self.bmi_label = tk.Label(master, text="BMI:")
#         self.bmi_label.grid(row=5, column=0)
#         self.bmi_entry = tk.Entry(master)
#         self.bmi_entry.grid(row=5, column=1)
#
#         self.diabetes_pedigree_function_label = tk.Label(master, text="Diabetes Pedigree Function:")
#         self.diabetes_pedigree_function_label.grid(row=6, column=0)
#         self.diabetes_pedigree_function_entry = tk.Entry(master)
#         self.diabetes_pedigree_function_entry.grid(row=6, column=1)
#
#         # Create the predict button
#         self.predict_button = tk.Button(master, text="Predict", command=self.predict)
#         self.predict_button.grid(row=7, column=0, columnspan=2)
#
#     def predict(self):
#         try:
#             input_data = {
#                 'Pregnancies': [int(self.pregnancies_entry.get())],
#                 'Glucose': [int(self.glucose_entry.get())],
#                 'BloodPressure': [int(self.blood_pressure_entry.get())],
#                 'SkinThickness': [int(self.skin_thickness_entry.get())],
#                 'Insulin': [int(self.insulin_entry.get())],
#                 'BMI': [float(self.bmi_entry.get())],
#                 'DiabetesPedigreeFunction': [
#                     float(self.diabetes_pedigree_function_entry.get())]
#             }
#             input_df = pd.DataFrame(input_data)
#             prediction = predict_diabetes(input_df)
#             messagebox.showinfo("Prediction", prediction)
#         except ValueError:
#             messagebox.showerror("Error", "Please enter valid values")
#
#
# # Start the GUI
# if __name__ == '__main__':
#     root = tk.Tk()
#     root.title("Diabetes Predictor")
#     app = App(root)
#     root.mainloop()

import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import self as self
from pywin.framework.app import App
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Load diabetes dataset
diabetes_data_path = os.path.join(os.path.dirname(__file__), 'diabetes_data.csv')
diabetes_data = pd.read_csv(diabetes_data_path)

# Split dataset into training and testing sets
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define neural network model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))

# Preprocess the input data
scaler = StandardScaler()
scaler.fit(X_train)
input_data = [[int(self.pregnancies_entry.get()), int(self.glucose_entry.get()), int(self.blood_pressure_entry.get()),
               int(self.skin_thickness_entry.get()), int(self.insulin_entry.get()), float(self.bmi_entry.get()),
               float(self.diabetes_pedigree_function_entry.get())]]
input_df = pd.DataFrame(input_data, columns=X_train.columns)
input_df = scaler.transform(input_df)

# Predict the input data using the trained model
prediction = model.predict_classes(input_df)
if prediction == 0:
    messagebox.showinfo("Prediction", "You do not have diabetes")
else:
    messagebox.showinfo("Prediction", "You have diabetes")

# Display descriptive analytics
diabetes_data.describe()
diabetes_data.hist()
diabetes_data.corr()

# Display predictive analytics
accuracy = model.evaluate(X_test, y_test)[1]
y_pred = model.predict_classes(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Display the ROC curve
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Neural Network')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Create the GUI
root = tk.Tk()
root.title("Diabetes Predictor")
app = App(root)
root.mainloop()

# Path: C964Capstone.py
# The code below is the code for the Diabetes Predictor app without the GUI.
# It is a simple app that takes in the values for the 7 features and predicts whether the patient has diabetes or not.
# The app is built using the scikit-learn library.
