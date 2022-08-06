from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import sketch

st.set_page_config(page_title="First prototype of a streamlit", page_icon="🔀")

st.write(
    """
# 🔀 Example search for joins from Sketch!
Upload a CSV file and find enrichment sources that can possibly be joined to it.
"""
)

apikey = st.text_input("API Key", value="8f79be2b6d0d47ccb8192e46f38c80ce")

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Data preview")
    st.dataframe(df.head())

    test = sketch.Portfolio.from_dataframe(df)
    html = test.find_joinables_html(apiKey=apikey, rawhtml=True)

    components.iframe(
        src=f"data:text/html;charset=utf-8,{quote(html)}", width=700, height=600
    )
