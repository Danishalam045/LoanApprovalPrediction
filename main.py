# main.py

# Import libraries
import joblib # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt# type: ignore
import seaborn as sns# type: ignore
from sklearn import preprocessing, metrics # type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.neighbors import KNeighborsClassifier# type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.svm import SVC# type: ignore
from sklearn.linear_model import LogisticRegression# type: ignore

# Load dataset
data = pd.read_csv("data/LoanApprovalPrediction.csv")
print(data.head())

# Drop Loan_ID as it's unique
data.drop(['Loan_ID'], axis=1, inplace=True)

# Visualize categorical values
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

# Plot categorical feature distributions
plt.figure(figsize=(18, 36))
index = 1
for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.tight_layout()
plt.show()

# Encode categorical variables
label_encoder = preprocessing.LabelEncoder()
for col in object_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Check for missing values and fill them
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
print(data.isna().sum())

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.show()

# Catplot
sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.show()

# Train-test split
X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)

# Model training
models = [
    RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7),
    KNeighborsClassifier(n_neighbors=3),
    SVC(),
    LogisticRegression()
]

# Train and test accuracy
for clf in models:
    clf.fit(X_train, Y_train)
    Y_pred_train = clf.predict(X_train)
    Y_pred_test = clf.predict(X_test)
    print(f"\n{clf.__class__.__name__}:")
    print(f"Train Accuracy: {100 * metrics.accuracy_score(Y_train, Y_pred_train):.2f}%")
    print(f"Test Accuracy:  {100 * metrics.accuracy_score(Y_test, Y_pred_test):.2f}%")


best_model = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
best_model.fit(X_train, Y_train)
joblib.dump(best_model, 'model.pkl')