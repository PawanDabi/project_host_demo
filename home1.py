import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import streamlit as st
st.set_page_config(page_title="Air Quality Pridiction . Streamlit",
    page_icon="https://cdn-icons-png.flaticon.com/512/3090/3090011.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
hide_menu = """
    <style>
    #MainMenu{
        visibility:hidden;
    }
    footer{
        visibility:hidden;
    }
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
st.sidebar.markdown("")
st.markdown("""
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: green;'>Industrial Air Quality</span>
        <span style='color: red;'>Analysis </span>
        <span style='color: green;'>& </span>
        <span style='color: red;'>Prediction. </span>
    </h1>
    """, unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center; font-size: 40px;'>
        <span style='color: white;'>Given Dataset...</span>
    </h1>
    """, unsafe_allow_html=True)
df=pd.read_csv('new_annual_conc_by_monitor_2024.csv')
if 'df' not in st.session_state:
    st.session_state.df = df
st.write(df)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center;'>Basic
        <span style='color: red;'> EDA Analysis</span>
    on Dataset</h1>""",
    unsafe_allow_html=True
)

tabs = st.tabs([
        "Data Types","Summary", "Features Names", "Meta Data (Info)",
        "Row/Column No.", "Numeric Data", "Categoric Data","Unique Values","Missing Values",
        "Missing Values Column", "Missing Values Percent.","Duplicate Rows"
    ])
with tabs[0]:
    left_side,right_side=st.columns((5,15))
    with left_side:
        st.write(":green[Column Names and Their Data Types:]")
        data_types = pd.DataFrame({
            "Data Type": df.dtypes
        })
        st.write(data_types)
        st.download_button(":green[Download Data Types as .CSV]",
        data_types.to_csv(),
        file_name='data_types.csv',
        mime='csv'
        )
    with right_side:
        st.write(":green[Summary of Data Types:]")
        type_summary = df.dtypes.value_counts().reset_index()
        type_summary.columns = ["Data Type", "Total Count"]
        st.write(type_summary)

with tabs[1]:
    left_side,right_side=st.columns((5,7))
    with left_side:
        st.write(":green[Summary of Numeric Dataset:]")
        numeric=['int32','int64','float32','float64']
        summary=df.describe(include=numeric).round(2)
        st.write(summary)
        st.download_button(":green[Download Summary as .CSV]",
        summary.to_csv(),
        file_name='Numeric_Summary.csv',
        mime='csv'
        )
    with right_side:
        st.write(":green[Summary of Categoric Dataset:]")
        Categoric=['object']
        summary=df.describe(include=Categoric)
        st.write(summary)
        st.download_button(":green[Download Summary as .CSV]",
        summary.to_csv(),
        file_name='categoric_summary.csv',
        mime='csv'
        )

with tabs[2]:
    st.write(":green[Column Names:]")
    st.write(df.columns.tolist())

with tabs[3]:
    st.write(":green[Meta Information about Dataset :]")
    with st.container(border=True):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    st.download_button(":green[Download Meta Data as .txt]",
    s,
    file_name='meta_data_information.txt',
    mime='txt/plan'
    )
with tabs[4]:
    st.write("No. of :green[Row] in DataSet: ")
    st.write(df.shape[0])
    st.write("No. of :green[Columns] in DataSet: ")
    st.write(df.shape[1])

with tabs[5]:
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    no_of_numeric = df.select_dtypes(include=numerics)
    st.write("No. of :green[Numerical] Columns: ")
    st.write(len(no_of_numeric.columns))
    right_col, left_col = st.columns((2, 8))
    with right_col:
        st.write(no_of_numeric.columns)
    with left_col:
        st.write(no_of_numeric)
        st.download_button(
            ":green[Download Data File as .CSV]",
            no_of_numeric.to_csv(),
            file_name='numeric_dataset.csv',
            mime='csv',
            key="slider009"
        )
with tabs[6]:
    categoric = ['object']
    no_of_categoric = df.select_dtypes(include=categoric)
    if no_of_categoric.empty:
        st.warning(":red[No Such Categorical Column Found in Dataset]")
    else:
        st.write("No. of :green[Categorical] Columns: ")
        st.write(len(no_of_categoric.columns))
        right_col, left_col = st.columns((2, 8))
        with right_col:
            st.write(no_of_categoric.columns)
        with left_col:
            st.write(no_of_categoric)
            st.download_button(
                ":green[Download Data File as .CSV]",
                no_of_categoric.to_csv(),
                file_name='categoric_dataset.csv',
                mime='csv',
                key="slider0009"
            )
with tabs[7]:
    unique_values = df.nunique()
    st.write(":green[Number of Unique Values per Column:]")
    st.write(unique_values)
    st.download_button(
        ":green[Download Data File as .CSV]",
        unique_values.to_csv(),
        file_name='unique_values.csv',
        mime='csv',
        key="slider00010"
    )

with tabs[8]:
    miss_col = df.columns[df.isnull().any()]
    if miss_col.empty:
        st.warning(":red[There is no Such Missing Records Found in the Dataset]")
    else:
        st.write("Columns Name that Contains :green[Missing Records : ]")
        st.write(len(miss_col))
        right_col,left_col=st.columns((2,8))
        with right_col:
            st.write(miss_col)
        with left_col:
            empty_dataset = df.isnull()
            st.dataframe(empty_dataset)
            st.download_button(":green[Download Data File as .CSV]",
            empty_dataset.to_csv(),
            file_name='missing_dataset.csv',
            mime='csv')
        st.markdown("---")
        numeric_data_col=df.select_dtypes(include=['float64', 'float32', 'int32', 'int64'])
        categoric_data_col=df.select_dtypes(include=['object'])
        right_col_numeric,left_col_categoric=st.columns((5,5))
        with right_col_numeric:
            numeric_column_with_missing_value=numeric_data_col.columns[numeric_data_col.isnull().any()]
            st.write(":green[Numerical] Columns Name that Contains Missing Records")
            st.write(numeric_column_with_missing_value)
            st.write(":green[Total: ]",len(numeric_column_with_missing_value))
        with left_col_categoric:
            final_categoric=categoric_data_col.columns[categoric_data_col.isnull().any()]
            st.write(":green[Categorical] Features Name that Contains Missing Records")
            st.write(final_categoric)
            st.write(":green[Total: ]",len(final_categoric))

with tabs[9]:
    missing_per_col = df.isnull().sum()
    st.write(":green[Missing Records per Features :]")
    st.write(missing_per_col)

with tabs[10]:
    percent_missing = df.isnull().sum() / len(df) * 100
    st.write("The following shows the percentage of :green[Missing Records] for each feature in dataset:")
    st.write(percent_missing)

with tabs[11]:
    duplicate_rows = df[df.duplicated()]
    if duplicate_rows.empty:
        st.write(":green[No Duplicate Rows Found in the Dataset]")
    else:
        st.write("Duplicate Rows in Dataset:")
        st.write(duplicate_rows)
        st.download_button(
            ":green[Download Duplicate Rows as .CSV]",
            duplicate_rows.to_csv(index=False),
            file_name='duplicate_rows.csv',
            mime='csv',
            key="duplicate_rows_download"
        )
st.sidebar.markdown("")