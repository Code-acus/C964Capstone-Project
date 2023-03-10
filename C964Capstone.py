import pandas as pd
import plotly.express as px
from scipy.stats._morestats import Mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data into a Pandas dataframe
data = pd.read_csv('diabetes_data.csv')

# If the file is in a different directory, provide the full path to the file
# data = pd.read_csv('/path/to/diabetes_data.csv')

# Data Wrangling
processed_df = data.drop(columns=['Unnamed: 0'])  # Removing unnecessary column
X = processed_df.drop(columns=['Outcome'])
y = processed_df['Outcome']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model to the training data
logreg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Evaluate the model's performance
accuracy = logreg.score(X_test, y_test)
print("Accuracy:", accuracy)

# Descriptive Method
# Data Exploration
fig1 = px.pie(processed_df, names='Outcome', title="Diabetes Outcome Distribution",
              color_discrete_sequence=["green", "red"], category_orders={'Outcome': [0, 1]})
fig1.show()

fig2 = px.scatter_matrix(processed_df,
                         dimensions=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                                     "DiabetesPedigreeFunction"],
                         color="Outcome", title="Scatter Matrix of Diabetes Data",
                         color_discrete_sequence=["green", "red"], category_orders={'Outcome': [0, 1]})
fig2.show()

fig3 = px.scatter_matrix(df,
    dimensions=["Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity"], color="Diagnosis",title="Worst Morphologies", color_discrete_sequence=["green","red"], category_orders={'Diagnosis': ['B', 'M']})
fig3.update_layout(
    width=950,
    height=950,
)
fig3.show()

# Data Inspection

px.scatter(df, x='Area', y='Worst Area', color='Diagnosis', title="Typical v.s. Worst Area Morphology",
              color_discrete_sequence=["green","red"], marginal_x="histogram", marginal_y="histogram")

# Data Exploration

fig1 = px.pie(df, names='Diagnosis', title="Benign(B) V.S. Malignant(M) Case Sampling",
              color_discrete_sequence=["green","red"], category_orders={'Diagnosis': ['B', 'M']})
fig1.show()

fig2 = px.scatter_matrix(df,
    dimensions=["Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness", "Concavity"],
                         color="Diagnosis",title="Typical Morphologies", color_discrete_sequence=["green","red"],
                         category_orders={'Diagnosis': ['B', 'M']})
fig2.update_layout(
    width=950,
    height=950,
)
fig2.show()

fig3 = px.scatter_matrix(df,
    dimensions=["Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity"], color="Diagnosis",title="Worst Morphologies", color_discrete_sequence=["green","red"], category_orders={'Diagnosis': ['B', 'M']})
fig3.update_layout(
    width=950,
    height=950,
)
fig3.show()

# Data Inspection

px.scatter(df, x='Area', y='Worst Area', color='Diagnosis', title="Typical v.s. Worst Area Morphology",
           color_discrete_sequence=["green","red"], marginal_x="hi")

# Non-Descriptive Method
# Machine Learning Algorithm
# In this section, the machine learning algorithm is created and trained using the data collected in the previous section.
# The algorithm is then tested and evaluated.

X_train, X_test, y_train, y_test = train_test_split(model.drop('M',axis=1),
                                                    model['M'], test_size=0.30,
                                                    random_state=123)

LogisticRegression(max_iter=600)

# Support Vector Machine

algorithm_2 = svm.SVC(gamma=1e-06, C=100000000.0)
algorithm_2.fit(X_train,y_train)
SVC(C=100000000.0, gamma=1e-06)

# Evaluating Accuracy

fna_diagnosis = algorithm.predict(X_test)
print(classification_report(y_test,fna_diagnosis))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, fna_diagnosis))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, fna_diagnosis))
print('Root-Mean-Square Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, fna_diagnosis)))
Mean Absolute Error (MAE): 13.444444444444445
Mean Squared Error (MSE): 0.07602339181286549
Root-Mean-Square Error (RMSE): 0.2757233972895037
cm_1 = metrics.confusion_matrix(y_test, fna_diagnosis, labels=algorithm.classes_)
disp_1 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_1, display_labels=algorithm.classes_)
disp_1.plot();

# Support Vector Machine

fna_diagnosis_2 = algorithm_2.predict(X_test)
print(classification_report(y_test,fna_diagnosis_2))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, fna_diagnosis_2))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, fna_diagnosis_2))
print('Root-Mean-Square Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, fna_diagnosis_2)))
Mean Absolute Error (MAE): 11.988304093567251
Mean Squared Error (MSE): 0.10526315789473684
Root-Mean-Square Error (RMSE): 0.3244428422615251
cm_2 = metrics.confusion_matrix(y_test, fna_diagnosis_2, labels=algorithm_2.classes_)
disp_2 = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=algorithm_2.classes_)
disp_2.plot();

# Conclusion
# Functionl Dashboard
# The following code is used to create the functional dashboard.

tab1 = HBox([input_container])
tab2 = VBox([graph_container_1])
tab3 = VBox([graph_container_2])
tab4 = VBox([graph_container_3])
tab = widgets.Tab(children=[tab1, tab2, tab3, tab4])
tab.set_title(0, 'Morphology Input')
tab.set_title(1, 'Reference Data')
tab.set_title(2, 'Test Contrasting')
tab.set_title(3, 'Accuracy Monitor')

application_dashboard = VBox(children=[tab])

# application_dashboard




