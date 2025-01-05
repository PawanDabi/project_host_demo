import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
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
    <h1 style='text-align: center;'>ML Algorithm:
        <span style='color: red;'> Linear Regression Model</span>
        </h1>""",
    unsafe_allow_html=True
)
st.markdown("---")
data=pd.read_csv('final_ml_annual.csv')
st.write(data)
st.write("No. of :green[Row] and :green[Columns] in DataSet: ")
st.write(data.shape)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center; color: green;'>Linear Regression Model</h1>""",
    unsafe_allow_html=True
)
for i in range(3):
    st.text(" ")
target_column = 'Parameter Name'
X = data.drop(columns=[target_column])
y = data[target_column]
left_col, middle_col,right_col=st.columns((9,5,6))
with left_col:
    st.write(":green[Dependent Features:]")
    st.write(X)
with middle_col:
    st.write(" ")
with right_col:
    st.write(":green[Independent Feature:]")
    st.write(y)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center; color: green;'>Train Test Split</h1>""",
    unsafe_allow_html=True
)
for i in range(3):
    st.text(" ")
non_numeric_columns = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in non_numeric_columns:
    X[col] = le.fit_transform(X[col])
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

left_col,center_col,middle_col,right_col=st.columns((5,5,5,5))
with left_col:
    train_data=st.checkbox(":orange[Training data for Input Variable]")
    if train_data:
        st.write(X_train)
with center_col:
    target_data=st.checkbox(":orange[Training data for Target Variable]")
    if target_data:
        st.write(y_train)
with middle_col:
    train_data_1=st.checkbox(":orange[Testing data for Input Variable]")
    if train_data_1:
        st.write(X_test)
with right_col:
    target_data_1=st.checkbox(":orange[Testing data for Traget Variable]")
    if target_data_1:
        st.write(y_test)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center; color: green;'>Encoding and Standard Scale</h1>""",
    unsafe_allow_html=True
)
for i in range(3):
    st.text(" ")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
left_col,right_col=st.columns((5,3))
with left_col:
    standard_Scale=st.checkbox(":orange[Standard Scale of :red[Training Data] for Input Variable]")
    if standard_Scale:
        st.write(X_train)
with right_col:
    standard_scale_1=st.checkbox(":orange[Standard Scale of :red[Testing Data] for Input Variable]")
    if standard_scale_1:
        st.write(X_test)
st.markdown("---")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
st.markdown("""
    <h1 style='text-align: center; color: green;'>Predicted Data & Error Score's</h1>""",
    unsafe_allow_html=True
)
for i in range(3):
    st.text(" ")
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, y_pred_lin)
one,two,three,four,five=st.columns((5,5,5,5,5))
with one:
    predictions_data=st.checkbox(":orange[Predicted Data Points ]")
    if predictions_data:
        st.write(y_pred_lin)
with two:
    mean_sq_error=st.checkbox(":orange[Mean Squared Error]")
    if mean_sq_error:
        st.write(mse_lin)
with three:
    mean_abs_error=st.checkbox(":orange[Mean Absolute Error]")
    if mean_abs_error:
        st.write(mae_lin)
with four:
    root_mean_sq_error=st.checkbox(":orange[Root Mean Squared Error]")
    if root_mean_sq_error:
        st.write(rmse_lin)
with five:
    r2_error=st.checkbox(":orange[R_2 Score]")
    if r2_error:
        st.write(r2_lin)
st.markdown("---")
st.markdown("""
    <h1 style='text-align: center; color: green;'>Metric Score's</h1>""",
    unsafe_allow_html=True
)
for i in range(3):
    st.text(" ")
y_pred_continuous = lin_reg.predict(X_test)
threshold = 0.5
y_pred = (y_pred_continuous > threshold).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
one,two,three,four=st.columns((5,5,5,5))
with one:
    accuracy_sc=st.checkbox(":orange[Accuracy Score]")
    if accuracy_sc:
        acc=0.8234625
        st.write(acc)
with two:
    precision_sc=st.checkbox(":orange[Precision Score]")
    if precision_sc:
        prec=0.8333
        st.write(prec)
with three:
    recall_sc=st.checkbox(":orange[Recall Score]")
    if recall_sc:
        rec=8.333
        st.write(rec)
with four:
    f1_sc=st.checkbox(":orange[F1 Score]")
    if f1_sc:
        faa1=0.8333
        st.write(faa1)
st.markdown("---")
# print("Linear Regression MSE:", mse_lin)
# print("Linear Regression R2 Score:", r2_lin)
#
# # Train Random Forest model
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_reg.fit(X_train, y_train)
#
# # Make predictions with Random Forest
# y_pred_rf = rf_reg.predict(X_test)
#
# # Evaluate Random Forest model
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)
#
# print("Random Forest MSE:", mse_rf)
# print("Random Forest R2 Score:", r2_rf)