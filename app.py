# Data processing #
import pandas as pd # Data processing - CSV file I/O.
import os, shutil, random, glob
import numpy as np # Linear algebra

# Data analysis & visualization #
import matplotlib #collection of functions for scientific and publication-ready visualization
import matplotlib.pyplot as plt # Show image

# Model evaluation #
from sklearn.utils import shuffle
from sklearn.utils import all_estimators
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# ROC Plot #
from sklearn.metrics import RocCurveDisplay

# Data modeling & evaluation #
from sklearn.tree import DecisionTreeClassifier

# Save & Loan Model #
#!pip install joblib
import joblib

#Install Streamlit library
!pip install -q streamlit

df_X_train = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_train.csv')
df_y_train = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_train.csv')
df_X_validate = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_validate.csv')
df_y_validate = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_validate.csv')
df_X_test = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_test.csv')
df_y_test = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_test.csv')

!wget https://raw.githubusercontent.com/Datamazing-My/CCP/main/CCP_01_DT.h5 -O CCP_01_DT.h5
clf_dt = joblib.load('CCP_01_DT.h5')

# Predict the classes on the test data.
y_pred = clf_dt.predict(df_X_test)
y_pred

# Predict the classes on the test data, and return the probabilities for each class
y_proba = clf_dt.predict_proba(df_X_test)
y_proba

print(confusion_matrix(df_y_test, y_pred))

print(classification_report(df_y_test, y_pred, labels=[0,1]))

RocCurveDisplay.from_predictions(df_y_test, y_proba[:, 1])
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.show()




