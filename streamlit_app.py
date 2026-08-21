import streamlit as st
import pandas as pd
st.title('MACHINE LEARNIG APP')

st.info('THIS IS AN MACHINE LERANING APP BUILDS ML MODELS')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df

  st.write('**x**')
  x = df.drop('species',axis=1)
  x

  st.write('**y**')
  y = df.species
  y

with st.expander('Data visualisation'):

  st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island',{'Torgersen','Biscoe','Dream'})
  gender = st.selectbox('Gender',{'male','female'})
  bill_length_mm = st.slider('Bill length {mm}',32.1,59.6,43.9)
  bill_depth_mm = st.slider('Bill Depth {mm}',13.1,21.5,17.2)
  flipper_length_mm = st.slider('Flipper lenghth {mm}',172.0,231.0,201.0)
  body_mass_g = st.slider('Body mass {g}',2700.0,6300,4207.0)

  data = {'island':island,
          'bill_length_mm':bill_length_mm,
          'bill_depth_mm':bill_depth_mm,
          'flipper_length_mm':flipper_length_mm,
          'body_mass_g':body_mass_g,
          'gender':gender}
  input_df = pd.DataFrame(dataindex = [0])
  input_penguins  = pd.concat([input_df,x],axis = 0))
with  st.expander('Input feautures'):
  st.write('**Input penguins**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins
