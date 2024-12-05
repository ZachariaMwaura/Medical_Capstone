import streamlit as st
import pandas as pd
import numpy as np

import pickle

# create title
st.title("Departments for patients")

my_description = st.text_area(label = "Patient Description Type Here", height = 300)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
model.predict([my_description])

st.markdown(f"#### The department for this patient is: {str(model.predict([my_description]))}")