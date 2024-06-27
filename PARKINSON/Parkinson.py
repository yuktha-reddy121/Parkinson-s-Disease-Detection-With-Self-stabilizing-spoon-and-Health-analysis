#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#DATA PRE-PROCESSING
data = pd.read_csv('Parkinsson disease.csv')
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#FEATURE SELECTION
    #DEFINING A FUNCTION TO IDENTIFY FEATURES BASED ON THEIR CORRELATION
def correlation(dataset, threshold):
  col_corr = set()
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        colname = corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr

corr_features = correlation(X_train,0.8)#here, correlation coefficient is chosen to be 0.8
print(corr_features)

X_train = X_train.drop(corr_features,axis=1) #dropping the features that satisfied the criteria
X_test = X_test.drop(corr_features,axis=1) #dropping the features that satisfied the criteria

c = list(X_train.columns.unique())
print (*c, sep ='\n ')#printing the remaining features


#TRAINING LOGISTIC REGRESSION MODEL
lr = LogisticRegression()
lr.fit(X_train,Y_train)
lr_pred = lr.predict(X_test)

#TRAINING DECISION TREE MODEL
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt_pred = dt.predict(X_test)

#TRAINING KNN MODEL
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
knn_pred = knn.predict(X_test)

#INTEGRATING ML MODEL WITH STREAMLIT PAGE
st.title('PARKINSONS DISEASE PREDICTOR')

with st.sidebar:
  option = option_menu('SELECT MODEL',options=['LOGISTIC REGRESSION','DECISION TREE','KNN CLASSIFIER'],default_index=0)

c1, c2, c3 = st.columns(3)
with c1:
  Fo = st.number_input('ENTER Fo(Hz)',step=.1)
  Fhi = st.number_input('ENTER Fhi(Hz)',step=.1)
  Flo = st.number_input('ENTER Fo(Hz)')
with c2:
  Jitter = st.number_input('ENTER MDVP : Jitter(%)')
  RPDE = st.number_input('ENTER RPDE : ')
  DFA = st.number_input('ENTER DFA : ')
with c3:
  spread1 = st.number_input('ENTER SPREAD1 : ')
  spread2 = st.number_input('ENTER SPREAD2 : ')
  D2 = st.number_input('ENTER D2 : ')

new = [[Fo,Fhi,Flo,Jitter,RPDE,DFA,spread1,spread2,D2]]

submit = st.button('SUBMIT')

if submit:
  if option == 'LOGISTIC REGRESSION':

    st.header('PREDICTION USING LOGISTIC REGRESSION MODEL : ')
    lr_res = lr.predict(new)
    if lr_res == 0:
      st.success('NO PARKINSONS DETECTED')
    else:
      st.error('PARKINSONS DETECTED')

    st.header('MODEL PARAMETERS')
    st.error("ACCURACY : {0}%".format(round(accuracy_score(Y_test, lr_pred), 2) * 100))
    st.warning("PRECISION : {0}%".format(round(precision_score(Y_test, lr_pred), 2) * 100))
    st.info("F1 SCORE  : {0}%".format(round(f1_score(Y_test, lr_pred), 2) * 100))
    st.success("R2 SCORE  : {0}%".format(round(recall_score(Y_test, lr_pred), 2) * 100))

    st.header('REPORT')
    report1 = classification_report(Y_test, lr_pred, target_names=['negative', 'positive'], output_dict=True)
    report1_df = pd.DataFrame(report1).transpose()
    st.dataframe(report1_df, height=212, width=1000)

    st.header('MODEL GRAPHS')
    sns.color_palette("Spectral", as_cmap=True)
    plt.style.use('fivethirtyeight')
    ax31 = sns.countplot(x='status', data=data, hue='status')
    st.pyplot(ax31.figure)

  elif option == 'DECISION TREE':
    st.header('PREDICTION USING DECISION TREE MODEL : : ')
    dt_res = dt.predict(new)
    if dt_res == 0:
      st.success('NO PARKINSONS DETECTED')
    else:
      st.error('PARKINSONS DETECTED')

    st.header('MODEL PARAMETERS')
    st.error("ACCURACY : {0}%".format(round(accuracy_score(Y_test, dt_pred), 2) * 100))
    st.warning("PRECISION : {0}%".format(round(precision_score(Y_test, dt_pred), 2) * 100))
    st.info("F1 SCORE  : {0}%".format(round(f1_score(Y_test, dt_pred), 2) * 100))
    st.success("R2 SCORE  : {0}%".format(round(recall_score(Y_test, dt_pred), 2) * 100))

    st.header('REPORT')
    report2 = classification_report(Y_test, dt_pred, target_names=['negative', 'positive'], output_dict=True)
    report2_df = pd.DataFrame(report2).transpose()
    st.dataframe(report2_df, height=212, width=1000)

    st.header('MODEL GRAPHS')
    sns.color_palette("Spectral", as_cmap=True)
    plt.style.use('fivethirtyeight')
    ax31 = sns.countplot(x='status', data=data, hue='status')
    st.pyplot(ax31.figure)

  else:
    st.header('PREDICTION USING KNN CLASSIFIER MODEL : ')
    knn_res = knn.predict(new)
    if knn_res == 0:
      st.success('NO PARKINSONS DETECTED')
    else:
      st.error('PARKINSONS DETECTED')

    st.header('MODEL PARAMETERS')
    st.error("ACCURACY : {0}%".format(round(accuracy_score(Y_test, knn_pred), 2) * 100))
    st.warning("PRECISION : {0}%".format(round(precision_score(Y_test, knn_pred), 2) * 100))
    st.info("F1 SCORE  : {0}%".format(round(f1_score(Y_test, knn_pred), 2) * 100))
    st.success("R2 SCORE  : {0}%".format(round(recall_score(Y_test, knn_pred), 2) * 100))

    st.header('REPORT')
    report3 = classification_report(Y_test, knn_pred, target_names=['negative', 'positive'], output_dict=True)
    report3_df = pd.DataFrame(report3).transpose()
    st.dataframe(report3_df, height=212, width=1000)

    st.header('MODEL GRAPHS')
    sns.color_palette("Spectral", as_cmap=True)
    plt.style.use('fivethirtyeight')
    ax31 = sns.countplot(x='status', data=data, hue='status')
    st.pyplot(ax31.figure)
  
  
else:
  st.error('CHECK ALL INPUTS & SUBMIT')
  
