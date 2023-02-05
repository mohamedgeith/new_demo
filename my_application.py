import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from category_encoders import OneHotEncoder
from xgboost import XGBClassifier

st.title('Bank Customer Churn Prediction')

st.image('https://miro.medium.com/max/786/1*RAeucVCKyFGXArObBsYnrw.webp')
Geography=st.selectbox('Geography',['France','Spain','Germany'])
Gender = st.selectbox('Gender',['Male','Female'])
Age = st.slider('Age',18,92)
Tenure=st.slider('Tenure',0,10)
Balance=st.slider('Balance',51000,175000)
NumOfProducts=st.selectbox('NumOfProducts',['1','2','3','4'])
HasCrCard=st.selectbox('HasCrCard',['0','1'])
IsActiveMember=st.selectbox('IsActiveMember',['0','1'])
EstimatedSalary=st.slider('EstimatedSalary',50000,200000)
new_age=st.selectbox('new_age',['youth','adult','very adult'])

### convert inputs to dataframe 

df_model=pd.DataFrame({'Geography':[Geography],'Gender':[Gender],'Age':[Age],'Tenure':[Tenure],'Balance':[Balance],'NumOfProducts':[NumOfProducts],'HasCrCard':[HasCrCard],'IsActiveMember':[IsActiveMember],'EstimatedSalary':[EstimatedSalary],'nerw_age':[new_age]})

## load onehotencoding


one_hot_2=pickle.load(open('one_hot_2.pkl', 'rb'))
X_new=one_hot_2.transform(df_model)
trans=pickle.load(open('trans.pkl', 'rb')) 
X_new=trans.transform(df_model)

pickle.dump(model_5, open("XGBoost_model.pkl", 'wb'))
final_model=pickle.load(open('XGBoost_model.pkl', 'rb'))


churn_prop = model_5.predict_proba(X_new)[0][1] * 100
st.markdown(f'## Probability of churn: {round(churn_prop, 2)} %')






