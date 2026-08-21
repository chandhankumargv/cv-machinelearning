import streamlit as st
import pandas as pd
st.title('MACHINE LEARNIG APP')

st.info('THIS IS AN MACHINE LERANING APP BUILDS ML MODELS')
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv")
df

