# Data processing #
import pandas as pd # Data processing - CSV file I/O.
import os, shutil, random, glob
import numpy as np # Linear algebra

# Data analysis & visualization #
#import matplotlib #collection of functions for scientific and publication-ready visualization
#import matplotlib.pyplot as plt # Show image

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
#!pip install -q streamlit

df_X_train = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_train.csv')
df_y_train = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_train.csv')
df_X_validate = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_validate.csv')
df_y_validate = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_validate.csv')
df_X_test = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/X_test.csv')
df_y_test = pd.read_csv('https://github.com/Datamazing-My/CCP/raw/main/y_test.csv')

#!wget https://raw.githubusercontent.com/Datamazing-My/CCP/main/CCP_01_DT.h5 -O CCP_01_DT.h5
clf_dt = joblib.load('CCP_01_DT.h5')

st.write("""
# This is a wed application that predict customer churn!
""")

st.sidebar.header('User Input Parameters')
def user_input_features():
  Gender = st.sidebar.slider('Gender', 0, 1, 0)
  Senior_Citizen = st.sidebar.slider('Senior Citizen', 0, 1, 0)
  Partner = st.sidebar.slider('Partner', 0, 1, 0)
  Dependents = st.sidebar.slider('Dependents', 0, 1, 0)
  Tenure_Months = st.sidebar.slider('Tenure Months', 0, 72, 0)
  Phone_Service = st.sidebar.slider('Phone Service', 0, 1, 0)
  Multiple_Lines = st.sidebar.slider('Multiple Lines', 0, 1, 0)
  Internet_Service = st.sidebar.slider('Internet Service', 0, 1, 0)
  Online_Security = st.sidebar.slider('Online Security', 0, 1, 0)
  Online_Backup = st.sidebar.slider('Online Backup', 0, 1, 0)
  Device_Protection = st.sidebar.slider('Device Protection', 0, 1, 0)
  Tech_Suppoprt = st.sidebar.slider('Tech Suppoprt', 0, 1, 0)
  Streaming_TV = st.sidebar.slider('Streaming TV', 0, 1, 0)
  Streaming_Movies = st.sidebar.slider('Streaming Movies', 0, 1, 0)
  Contract = st.sidebar.slider('Contract', 0, 2, 0)
  Paperless_Billing = st.sidebar.slider('Paperless Billing', 0, 1, 0)
  Payment_Method = st.sidebar.slider('Payment Method', 0, 3, 0)
  Monthly_Charges = st.sidebar.slider('Monthly Charges', 18.25, 118.75, 18.25)
  Total_Charges = st.sidebar.slider('Total Charges', 0.00, 8684.80, 0.00)
  data = {'Gender': Gender,
          'Senior Citizen': Senior_Citizen,
          'Partner': Partner,
          'Dependents': Dependents,
          'Tenure Months': Tenure_Months,
          'Phone Service': Phone_Service,
          'Multiple Lines': Multiple_Lines,
          'Internet Service': Internet_Service,
          'Online Security': Online_Security,
          'Online Backup': Online_Backup,
          'Device Protection': Device_Protection,
          'Tech Suppoprt': Tech_Suppoprt,
          'Streaming TV': Streaming_TV,
          'Streaming Movies': Streaming_Movies,
          'Contract': Contract,
          'Paperless Billing': Paperless_Billing,
          'Payment Method': Payment_Method,
          'Monthly Charges': Monthly_Charges,
          'Total Charges': Total_Charges}
  features = pd.DataFrame(data, index=[0])
  return features

input_df = user_input_features()
st.subheader('User Input Parameters')
st.write(input_df)



