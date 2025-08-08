import os
import gc
import json
import time
import shutil
import threading
from io import StringIO

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dark5A9_r = sns.color_palette("dark:#5A9_r", as_cmap=True)

import datetime
from configurations.datetime_format import datetime_mapping, dt_to_system_format

import handler
from configurations.tuning_parameters import get_tune_parameters

import streamlit as st
def set_state_without_rerun(key, value):
    if key not in st.session_state or st.session_state[key] != value:
        st.session_state[key] = value


st.set_page_config(
    page_title="AutoML Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialise_sessionState():
    st.session_state.df = None if 'df' not in st.session_state else st.session_state.df
    st.session_state.preprocessing_steps = [] if 'preprocessing_steps' not in st.session_state else st.session_state.preprocessing_steps
    st.session_state.page = "Upload Data" if 'page' not in st.session_state else st.session_state.page
    st.session_state.config = {} if 'config' not in st.session_state else st.session_state.config
    st.session_state.feature_columns = [] if 'feature_columns' not in st.session_state else st.session_state.feature_columns
    st.session_state.label_column = None if 'label_column' not in st.session_state else st.session_state.label_column
    st.session_state.datetime_column = [] if 'datetime_column' not in st.session_state else st.session_state.datetime_column
    st.session_state.partition_columns = [] if 'partition_columns' not in st.session_state else st.session_state.partition_columns
    st.session_state.ml_task = None if 'ml_task' not in st.session_state else st.session_state.ml_task
    st.session_state.algorithm = None if 'algorithm' not in st.session_state else st.session_state.algorithm
    st.session_state.tune = "Auto" if 'tune' not in st.session_state else st.session_state.tune
    st.session_state.manual_params = {} if 'manual_params' not in st.session_state else st.session_state.manual_params
    st.session_state.split_method = "Percentage Split" if 'split_method' not in st.session_state else st.session_state.split_method
    st.session_state.dt_in_user_format = None if 'dt_in_user_format' not in st.session_state else st.session_state.dt_in_user_format
    st.session_state.datetime_format = None if 'datetime_format' not in st.session_state else st.session_state.datetime_format
    st.session_state.train_test_split = "80:20" if 'train_test_split' not in st.session_state else st.session_state.train_test_split
initialise_sessionState()

def clear_sessionState():
    st.session_state.df = None
    st.session_state.preprocessing_steps = []
    st.session_state.page = "Upload Data"
    st.session_state.config = {}
    st.session_state.feature_columns = []
    st.session_state.label_column = None
    st.session_state.datetime_column = []
    st.session_state.partition_columns = []
    st.session_state.ml_task = None
    st.session_state.algorithm = None
    st.session_state.tune = "Auto"
    st.session_state.manual_params = {}
    st.session_state.split_method = "Percentage Split"
    st.session_state.dt_in_user_format = None
    st.session_state.datetime_format = None
    st.session_state.train_test_split = "80:20"

def save_configuration():
    config = {
        "df": st.session_state.get("df"),
        "ml_task": st.session_state.get("ml_task"),
        "algorithm": st.session_state.get("algorithm"),
        "feature_columns": st.session_state.get("feature_columns", []),
        "label_column": st.session_state.get("label_column"),
        "datetime_column": st.session_state.get("datetime_column", None),
        "partition_columns": st.session_state.get("partition_columns", []),
        "preprocessing_steps": st.session_state.get("preprocessing_steps", []),
        "train_test_split": st.session_state.get("train_test_split", "80:20"),
        "tune": st.session_state.get("tune", "Auto"),
        "manual_params": st.session_state.get("manual_params", {}),
        "split_method": st.session_state.get("split_method", "Percentage Split"),
        "train_start_date": st.session_state.get("train_start_date"),
        "train_end_date": st.session_state.get("train_end_date"),
        "test_start_date": st.session_state.get("test_start_date"),
        "test_end_date": st.session_state.get("test_end_date"),
        "forecast_method": st.session_state.get("forecast_method", None),
        "forecast_start_date": st.session_state.get("forecast_start_date", None),
        "forecast_end_date": st.session_state.get("forecast_end_date", None),
        "forecast_periods": st.session_state.get("forecast_periods", None),
        "dt_in_user_format": st.session_state.get("dt_in_user_format"),
        "datetime_format": st.session_state.get("datetime_format"),
        "config_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.config = config
    st.success("Configuration saved successfully!")
    st.json(config)

def add_styling():
    st.markdown(
        """
        <style>
            div[role="radiogroup"] {
                flex-direction: column;
                gap: 8px;
            }
            
            input[type="radio"] + div {
                background: #131720;
                color: #FAFAFA;
                border-radius: 10px;
                border: 2px solid #262730;
                padding: 8px 18px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            input[type="radio"][tabindex="0"] + div {
                color: #FF4B4B;
                border: 2px solid #FF4B4B;
            }
            
            input[type="radio"][tabindex="0"] + div p {
                color: #FAFAFA;
            }

            div[role="radiogroup"] label > div:first-child {
                display: none;
            }

            div[role="radiogroup"] label {
                margin-right: 0px;
                cursor: pointer;
            }

            div[role="radiogroup"] {
                gap: 12px;
            }
            
            input[type="radio"]:not(:checked) + div:hover {
                box-shadow: 0 0 8px #FF4B4B;
                border-color: #41444C;
                background: #FF4B4B;
                color: #FAFAFA;
            }
            
            div.stSelectbox > div[data-baseweb="select"] {
                cursor: pointer;
            }
            
            div.stMultiSelect {
                cursor: pointer;
            }
            
            div[role="radiogroup"] label {
                display: block;
                width: 100%;
            }
            
            .stCheckbox {
                cursor: pointer;
            }
            .stCheckbox > label {
                display: block;
                padding: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <style>
        .custom-line-margin {
            margin-bottom: 5px;
        }
        .custom-paragraph-margin {
            margin-top: 15px;
        }
        .custom-tab-margin {
            margin-left: 25px;
        }
        </style>
    """, unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("AutoML Playground")
# add_styling()
# st.session_state.page = st.sidebar.radio(
#     "", 
#     ["Upload Data", "Data Overview", "Feature Engineering", "Model Configuration", "Summary"],
#     index=["Upload Data", "Data Overview", "Feature Engineering", "Model Configuration", "Summary"].index(st.session_state.page)
# )
add_styling()
page_options = ["Upload Data", "Data Overview", "Feature Engineering", "Model Configuration", "Summary", "Progress History"]
selected_page = st.sidebar.radio(
    "Navigation", 
    page_options,
    index=page_options.index(st.session_state.page) if st.session_state.page in page_options else 0,
    key="navigation_radio"
)
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()


# Page: Upload Data
if st.session_state.page == "Upload Data":
    st.header("Load Data*")
    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        data_input_method = st.radio("Select Data input method*", ["Enter File Path", "Upload CSV/Excel File"])
    if data_input_method == "Upload CSV/Excel File":
        uploaded_file = st.file_uploader("Choose a file*", type=["csv", "xls", "xlsx"])
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success(f"File uploaded successfully! Detected {len(st.session_state.df)} records.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        file_path = st.text_input("Enter Full File Path*")
        if file_path and st.button("Load Data"):
            file_path = file_path[1:-1] if (file_path.startswith(("\"", "\'")) and file_path.endswith(("\"", "\'"))) else file_path
            try:
                if file_path.endswith('.csv'):
                    st.session_state.df = pd.read_csv(file_path)
                elif file_path.endswith((".xlsx", ".xls")):
                    st.session_state.df = pd.read_excel(file_path)
                else:
                    st.error("Unsupported file format. Please use CSV or Excel files.")
                    st.session_state.df = None
                
                if st.session_state.df is not None:
                    st.success(f"File loaded successfully! Detected {len(st.session_state.df)} records.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        if st.button("Continue"):
            st.session_state.page = "Data Overview"
            st.rerun()


# Page: Data Overview
elif st.session_state.page == "Data Overview":
    st.header("Data Overview")
    if st.session_state.df is None:
        st.warning("Please first upload data in the 'Upload Data' section.")
    else:
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of rows: {len(st.session_state.df)}")
        with col2:
            st.write(f"Number of columns: {len(st.session_state.df.columns)}")
        
        unique_dtypes = st.session_state.df.dtypes.unique()
        st.write(f"<div class='.custom-line-margin'>{len(unique_dtypes)} Unique Datatypes:</div>", unsafe_allow_html=True)
        st.write("\n".join([f"<div class='custom-tab-margin'>{i}. {dtype}</div>" for i, dtype in enumerate(unique_dtypes, 1)]), unsafe_allow_html=True)
        st.subheader("Column Datatypes")
        dtypes_df = pd.DataFrame({
            "Column": st.session_state.df.columns,
            "Datatypes": st.session_state.df.dtypes.values.astype(str),
            "Missing Values": st.session_state.df.isna().sum().values,
            "Unique Values": st.session_state.df.nunique().values
        })
        st.dataframe(dtypes_df)
        st.subheader("Correlation Matrix")
        try:
            numeric_df = st.session_state.df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2", cmap=dark5A9_r, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns found for correlation analysis.")
        except Exception as e:
            st.error(f"Error generating correlation matrix: {e}")
        
        if st.button("Continue"):
            st.session_state.page = "Feature Engineering"
            st.rerun()


# Page: Feature Engineering
elif st.session_state.page == "Feature Engineering":
    st.header("Feature Engineering")
    if st.session_state.df is None:
        st.warning("Please upload data first in the Data Input section.")
    else:
        st.write(f"Data Preview")
        st.dataframe(st.session_state.df.head())
        st.write(f"... Total {len(st.session_state.df)} records")
        st.subheader("Feature(s)/Label Selection*")
        col1, col2 = st.columns(2)
        with col1:
            all_columns = st.session_state.df.columns.to_list()
            st.session_state.feature_columns = st.multiselect(
                "Select Feature Columns (At least 1; Optional for TimeSeries Forecasting)", 
                all_columns,
                default=st.session_state.feature_columns,
                key="feature_cols_multiselect"
            )
            st.session_state.partition_columns = st.multiselect(
                "Select Partition/GroupBy Columns (Optional):",
                all_columns,
                default=st.session_state.partition_columns,
                key="partition_cols_multiselect"
            )
        with col2:
            label_columns = list(set(all_columns) - set(st.session_state.feature_columns) - set(st.session_state.partition_columns))
            st.session_state.label_column = st.selectbox(
                "Select label column*",
                label_columns,
                index=label_columns.index(st.session_state.label_column) if st.session_state.label_column in label_columns else 0
            )
            datetime_options = [col for col in all_columns if st.session_state.df[col].dtype == "datetime64[ns]"]
            st.session_state.datetime_column = st.selectbox(
                "Select datetime column (Optional; but required for Time-Series Forecasting)",
                ["None"] + datetime_options + ["Explore Object Column"],
                index=datetime_options.index(st.session_state.datetime_column)+1 if st.session_state.datetime_column in datetime_options else 0
            )
            if st.session_state.datetime_column in datetime_options:
                st.write(f"<div class='.custom-line-margin'>Earliest Date: {st.session_state.df[st.session_state.datetime_column].min()}</div>", unsafe_allow_html=True)
                st.write(f"<div class='.custom-line-margin'>Latest Date: {st.session_state.df[st.session_state.datetime_column].max()}</div>", unsafe_allow_html=True)
                timeintervalFreq = pd.DatetimeIndex(st.session_state.df[st.session_state.datetime_column]).inferred_freq
                if not timeintervalFreq:
                    timeintervalFreq = st.session_state.df[st.session_state.datetime_column].diff().mode()[0]
                st.write(f"DateTime Interval Frequency/Bucket: {timeintervalFreq}")
            if st.session_state.datetime_column == "Explore Object Column":
                st.session_state.datetime_column = st.selectbox(
                    "Select column to act as datetime column:",
                    list(set(all_columns) - set(datetime_options)),
                    index=datetime_options.index(st.session_state.datetime_column) if st.session_state.datetime_column in datetime_options else len(datetime_options)
                )
                st.session_state.dt_in_user_format = st.selectbox(
                    "Select datetime format:",
                    ["custom"] + list(datetime_mapping.keys()),
                    index=0
                )
                if st.session_state.dt_in_user_format == "custom":
                    st.session_state.dt_in_user_format = st.text_input("Enter custom datetime format (case-sensitive):")
                    st.session_state.datetime_format = dt_to_system_format(st.session_state.dt_in_user_format)
                else:
                    st.session_state.datetime_format = datetime_mapping[st.session_state.dt_in_user_format]
                st.write(st.session_state.datetime_format)
                if st.button("Apply Date Format"):
                    try:
                        st.session_state.df[st.session_state.datetime_column] = pd.to_datetime(st.session_state.df[st.session_state.datetime_column], format=st.session_state.datetime_format)
                        st.success(f"Date format applied successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.subheader("Data Sorting")
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_column = st.selectbox("Sort by column:", ["None"] + all_columns)
        with sort_col2:
            if sort_column != "None":
                sort_order = st.selectbox("Sort Order:", ["Ascending", "Descending"])
                if st.button("Apply Sorting"):
                    order = True if sort_order == "Ascending" else False
                    st.session_state.df = st.session_state.df.sort_values(by=sort_column, ascending=order)
                    st.session_state.preprocessing_steps.append(f"Sorted data by {sort_column} in {sort_order} order.")
                    st.success(f"Data sorted by {sort_column} in {sort_order} order.")
                    st.rerun()
        
        if st.button("Randomise Data"):
            st.session_state.df = st.session_state.df.sample(frac=1).reset_index(drop=True)
            st.session_state.preprocessing_steps.append("Randomised data.")
            st.success("Data randomised successfully.")
            time.sleep(2)
            st.rerun()
        
        st.subheader("Missing Values Handling")
        missing_values_fill_method = st.selectbox(
            "Select method:",
            ["None", "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom/constant value", "Interpolate"]
        )
        if missing_values_fill_method != "None":
            if missing_values_fill_method == "Fill with custom/constant value":
                fill_value = st.text_input("Enter fill value:")
                if st.button("Apply Missing Values Handling"):
                    st.session_state.df = st.session_state.df.fillna(float(fill_value) if fill_value.replace(".", "", 1).isdigit() else fill_value)
                    st.session_state.preprocessing_steps.append(f"Filled missing values with {fill_value}.")
                    st.success(f"Missing Values Filled with custom value: {fill_value}")
                    time.sleep(2)
                    st.rerun()
            elif st.button("Apply Missing Values Treatment"):
                if missing_values_fill_method == "Drop rows":
                    st.session_state.df = st.session_state.df.dropna()
                    st.session_state.preprocessing_steps.append("Dropped rows with missing values.")
                    st.success("Rows with missing values dropped.")
                elif missing_values_fill_method == "Fill with mean":
                    for col in st.session_state.df.select_dtypes(include=[np.number]).columns:
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mean())
                    st.session_state.preprocessing_steps.append("Filled numeric missing values with mean")
                    st.success(f"Missing Values Filled using mean.")
                elif missing_values_fill_method == "Fill with median":
                    for col in st.session_state.df.select_dtypes(include=[np.number]).columns:
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].median())
                    st.session_state.preprocessing_steps.append("Filled numeric missing values with median")
                    st.success(f"Missing Values Filled using median.")
                elif missing_values_fill_method == "Fill with mode":
                    for col in st.session_state.df.select_dtypes(include=[np.number]).columns:
                        st.session_state.df[col] = st.session_state.df[col].fillna(st.session_state.df[col].mode()[0])
                    st.session_state.preprocessing_steps.append("Filled numeric missing values with mode")
                    st.success(f"Missing Values Filled using mode.")
                elif missing_values_fill_method == "Interpolate":
                    st.session_state.df = st.session_state.df.interpolate(method="linear", limit_direction="both")
                    st.session_state.preprocessing_steps.append("Interpolated missing values.")
                    st.success(f"Missing Values Filled using linear interpolation.")
                time.sleep(2)
                st.rerun()
        
        st.subheader("Feature Scaling")
        scaling_columns = st.multiselect(
            "Select columns for scaling (Optional):", 
            [col for col in st.session_state.df.columns if pd.api.types.is_numeric_dtype(st.session_state.df[col])]
        )
        if scaling_columns:
            scaling_method = st.selectbox("Select scaling method:", ["Min-Max Normalisation", "Standarisation (Z-Score)"])
            
            if st.button("Apply Scaling"):
                if scaling_method == "Min-Max Normalisation":
                    for col in scaling_columns:
                        min_val = st.session_state.df[col].min()
                        max_val = st.session_state.df[col].max()
                        st.session_state.df[col] = (st.session_state.df[col] - min_val) / (max_val - min_val)
                    st.session_state.preprocessing_steps.append(f"Applied Min-Max Normalisation to {', '.join(scaling_columns)}")
                else:
                    for col in scaling_columns:
                        mean_val = st.session_state.df[col].mean()
                        std_val = st.session_state.df[col].std()
                        st.session_state.df[col] = (st.session_state.df[col] - mean_val) / std_val
                    st.session_state.preprocessing_steps.append(f"Applied Standarisation to {', '.join(scaling_columns)}")
                st.success(f"Applied {scaling_method} to {', '.join(scaling_columns)}")
                time.sleep(2)
                st.rerun()
        if st.button("Continue"):
            st.session_state.page = "Model Configuration"
            st.rerun()


# Model Configuration Page
elif st.session_state.page == "Model Configuration":
    st.header("Model Configuration")
    
    if st.session_state.df is None:
        st.warning("Please upload data first in the Data Input section.")
    else:
        st.subheader("Select Machine Learning Task")
        ml_task_options = ["Regression", "Outlier Detection", "Time Series Forecasting", "Classification"]
        st.session_state.ml_task = st.selectbox(
            "Machine Learning Task:",
            ml_task_options,
            index=ml_task_options.index(st.session_state.ml_task) if st.session_state.ml_task in ml_task_options else 0
        )

        st.subheader("Train-Test Split Configuration")
        split_col1, split_col2 = st.columns(2)
        with split_col1:
            st.session_state.split_method = st.radio(
                "Select Split Method:",
                ["Percentage Split", "Time-based Split"]
            )
        with split_col2:
            if st.session_state.split_method == "Percentage Split":
                train_split, test_split = 80, 20
                train_split = st.slider(
                    "Select Train Set Percentage:", 
                    min_value = 50,
                    max_value = 90,
                    value = int(train_split),
                    step = 5
                )
                test_split = st.slider(
                    "Select Test Set Percentage:", 
                    min_value = 10,
                    max_value = 50,
                    value = 100 - int(train_split),
                    step = 5
                )
                st.session_state.train_test_split = f"{train_split}:{test_split}"
            elif st.session_state.split_method == "Time-based Split":
                if st.session_state.datetime_column == "None":
                    st.warning("Please select a datetime column for time-based split.")
                else:
                    try:
                        if not pd.api.types.is_datetime64_dtype(st.session_state.df[st.session_state.datetime_column]):
                            st.session_state.df[st.session_state.datetime_column] = pd.to_datetime(st.session_state.df[st.session_state.datetime_column])
                        
                        min_date = st.session_state.df[st.session_state.datetime_column].min()
                        max_date = st.session_state.df[st.session_state.datetime_column].max()
                        date_range = (max_date - min_date).days
                        if 'train_start_date' not in st.session_state:
                            st.session_state.train_start_date = min_date
                        if 'train_end_date' not in st.session_state:
                            st.session_state.train_end_date = min_date + pd.Timedelta(days=int(date_range * 0.8))
                        if 'test_start_date' not in st.session_state:
                            st.session_state.test_start_date = st.session_state.train_end_date + pd.Timedelta(days=1)
                        if 'test_end_date' not in st.session_state:
                            st.session_state.test_end_date = max_date
                        
                        st.subheader("Train-Test Date Ranges")
                        
                        st.write("**Training Period Selection:**")
                        train_col1, train_col2 = st.columns(2)
                        with train_col1:
                            train_start_date = st.date_input(
                                "Training Start Date:",
                                value=st.session_state.train_start_date.date(),
                                min_value=min_date.date(),
                                max_value=st.session_state.train_end_date.date()
                            )
                            st.session_state.train_start_date = pd.Timestamp(train_start_date)
                        
                        with train_col2:
                            train_end_date = st.date_input(
                                "Training End Date:",
                                value=st.session_state.train_end_date.date(),
                                min_value=st.session_state.train_start_date.date(),
                                max_value=max_date.date()
                            )
                            st.session_state.train_end_date = pd.Timestamp(train_end_date)
                        
                        train_range_percent = ((st.session_state.train_end_date - st.session_state.train_start_date).days / date_range) * 100
                        st.progress(train_range_percent / 100)
                        st.caption(f"Selected training range: {st.session_state.train_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.train_end_date.strftime('%Y-%m-%d %H:%M:%S')} ({train_range_percent:.1f}% of data)")
                        
                        st.write("**Testing Period Selection:**")
                        test_col1, test_col2 = st.columns(2)
                        with test_col1:
                            test_start_date = st.date_input(
                                "Testing Start Date:",
                                value=max(st.session_state.train_end_date, st.session_state.test_start_date).date(),
                                min_value=st.session_state.train_start_date.date(),
                                max_value=max_date.date()
                            )
                            st.session_state.test_start_date = pd.Timestamp(test_start_date)
                        
                        with test_col2:
                            test_end_date = st.date_input(
                                "Testing End Date:",
                                value=st.session_state.test_end_date.date(),
                                min_value=st.session_state.test_start_date.date(),
                                max_value=max_date.date()
                            )
                            st.session_state.test_end_date = pd.Timestamp(test_end_date)
                        
                        test_range_percent = ((st.session_state.test_end_date - st.session_state.test_start_date).days / date_range) * 100
                        st.progress(test_range_percent / 100)
                        st.caption(f"Selected testing range: {st.session_state.test_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.test_end_date.strftime('%Y-%m-%d %H:%M:%S')} ({test_range_percent:.1f}% of data)")
                        
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.set_xlim(min_date, max_date + pd.Timedelta(days=date_range*0.1))  # Add extra space for forecast
                        ax.axvspan(min_date, max_date, color='lightgray', alpha=0.5)  # Full dataset
                        ax.axvspan(st.session_state.train_start_date, st.session_state.train_end_date, color='lightblue', alpha=0.7)  # Training
                        ax.axvspan(st.session_state.test_start_date, st.session_state.test_end_date, color='lightgreen', alpha=0.7)  # Testing
                        ax.set_yticks([])
                        ax.set_xlabel('Date')
                        plt.tight_layout()
                        if st.session_state.ml_task != "Time Series Forecasting":
                            st.pyplot(fig)
                        
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown('<div style="background-color:lightblue;padding:5px;border-radius:5px;">Training Data</div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown('<div style="background-color:lightgreen;padding:5px;border-radius:5px;">Testing Data</div>', unsafe_allow_html=True)
                                
                    except Exception as e:
                        st.error(f"Error processing datetime column: {e}")
                        st.info("Please select a valid datetime column in the Feature Engineering section.")
            if st.session_state.ml_task == "Time Series Forecasting":
                try:
                    min_date = st.session_state.df[st.session_state.datetime_column].min()
                    max_date = st.session_state.df[st.session_state.datetime_column].max()
                    date_range = (max_date - min_date).days
                    
                    if 'forecast_method' not in st.session_state:
                        st.session_state.forecast_method = "Date Range"
                    if 'forecast_start_date' not in st.session_state:
                        st.session_state.forecast_start_date = max_date + pd.Timedelta(days=1)
                    if 'forecast_end_date' not in st.session_state:
                        st.session_state.forecast_end_date = max_date + pd.Timedelta(days=30)
                    if 'forecast_periods' not in st.session_state:
                        st.session_state.forecast_periods = 10
                    if 'forecast_period_unit' not in st.session_state:
                        st.session_state.forecast_period_unit = "Days"
                    
                    st.write("**Forecasting Period Selection:**")
                    st.session_state.forecast_method = st.radio(
                        "Forecast method:", 
                        ["Date Range", "Next N Periods"],
                        index=0 if st.session_state.forecast_method == "Date Range" else 1
                    )
                    
                    if st.session_state.forecast_method == "Date Range":
                        forecast_col1, forecast_col2 = st.columns(2)
                        with forecast_col1:
                            forecast_start_date = st.date_input(
                                "Forecast Start Date:",
                                value=st.session_state.forecast_start_date.date(),
                                min_value=st.session_state.test_end_date.date(),
                                max_value=(st.session_state.test_end_date + pd.Timedelta(days=365)).date()
                            )
                            st.session_state.forecast_start_date = pd.Timestamp(forecast_start_date)
                        
                        with forecast_col2:
                            forecast_end_date = st.date_input(
                                "Forecast End Date:",
                                value=st.session_state.forecast_end_date.date(),
                                min_value=st.session_state.forecast_start_date.date(),
                                max_value=(st.session_state.forecast_start_date + pd.Timedelta(days=365)).date()
                            )
                            st.session_state.forecast_end_date = pd.Timestamp(forecast_end_date)
                            
                        forecast_days = (st.session_state.forecast_end_date - st.session_state.forecast_start_date).days + 1
                        st.info(f"Forecasting for {forecast_days} days from {st.session_state.forecast_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.forecast_end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.session_state.forecast_periods = st.number_input(
                            "Number of periods to forecast:", 
                            min_value=1,
                            value=st.session_state.forecast_periods
                        )

                        
                        forecast_unit = st.selectbox(
                            "Forecast period unit:",
                            ["Days", "Weeks", "Months"]
                        )
                        st.session_state.forecast_period_unit = forecast_unit
                        
                        if forecast_unit == "Days":
                            forecast_end = st.session_state.test_end_date + pd.Timedelta(days=st.session_state.forecast_periods)
                        elif forecast_unit == "Weeks":
                            forecast_end = st.session_state.test_end_date + pd.Timedelta(weeks=st.session_state.forecast_periods)
                        else:
                            forecast_end = st.session_state.test_end_date + pd.Timedelta(days=st.session_state.forecast_periods*30)  # Approximation
                            
                        st.info(f"Forecasting for next {st.session_state.forecast_periods} {forecast_unit.lower()} (until approximately {forecast_end.strftime('%Y-%m-%d %H:%M:%S')})")

                    if st.session_state.split_method == "Time-based Split":
                        if st.session_state.forecast_method == "Date Range":
                            ax.axvspan(st.session_state.forecast_start_date, st.session_state.forecast_end_date, color='lightsalmon', alpha=0.7)  # Forecast
                        else:
                            ax.axvspan(st.session_state.test_end_date, forecast_end, color='lightsalmon', alpha=0.7)  # Forecast
                    
                        ax.set_yticks([])
                        ax.set_xlabel('Date')
                        plt.tight_layout()
                        st.pyplot(fig)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div style="background-color:lightblue;padding:5px;border-radius:5px;">Training Data</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div style="background-color:lightgreen;padding:5px;border-radius:5px;">Testing Data</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div style="background-color:lightsalmon;padding:5px;border-radius:5px;">Forecast Period</div>', unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"Error processing forecast period: {e}")
                    st.info("Please select a valid datetime column in the Feature Engineering section.")

        st.subheader("Select Algorithm")
        if st.session_state.ml_task == "Time Series Forecasting":
            algorithm_options = ["ARIMA", "SARIMA", "Prophet", "Neural Prophet", "Attention-Based LSTM", "XGBoost Forecaster"]
        elif st.session_state.ml_task == "Regression":
            algorithm_options = ["XGBoost Regressor", "Random Forest Regressor", "Linear Regression", "SVM Regressor"]
        elif st.session_state.ml_task == "Outlier Detection":
            algorithm_options = ["CatBoost Outlier Detector", "Local Outlier Factor", "One-Class SVM"]
        elif st.session_state.ml_task == "Classification":
            algorithm_options = ["XGBoost Classifier", "Logistic Regression", "Linear SVM Classifier"]
        else:
            algorithm_options = []
        st.session_state.algorithm = st.selectbox(
            "Select Algorithm:",
            algorithm_options,
            index=algorithm_options.index(st.session_state.algorithm) if st.session_state.algorithm in algorithm_options else 0
        )
        tune_col1, tune_col2 = st.columns(2)
        with tune_col1:
            if st.button("Manual Parameter Selection"):
                st.session_state.tune = "Manual"
        with tune_col2:
            if st.button("Automatic Parameter Selection"):
                st.session_state.tune = "Auto"
        
        parameters = get_tune_parameters(st.session_state.algorithm)
        if st.session_state.tune == "Auto":
            st.write(f"<div class='.custom-line-margin'>Hyperparameter Tuning Method -> Bayesian Optimisation will be used for {st.session_state.algorithm} algorithm on the following parameters and ranges:</div>", unsafe_allow_html=True)
            for parameter in parameters.keys():
                if parameters[parameter][-1] == "category":
                    st.write(f"<div class='.custom-line-margin'>{parameter}: {parameters[parameter][0]}<div>", unsafe_allow_html=True)
                elif parameters[parameter][-1] == "int" or parameters[parameter][-1] == "float":
                    stepText = f"at {parameters[parameter][2]} increment" if parameters[parameter][2] is not None else ""
                    st.write(f"<div class='.custom-line-margin'>{parameter}: {parameters[parameter][0]} to {parameters[parameter][1]} {stepText}</div>", unsafe_allow_html=True)
        else:
            if 'previous_algorithm' not in st.session_state or st.session_state.previous_algorithm != st.session_state.algorithm:
                st.session_state.manual_params = {}
                st.session_state.previous_algorithm = st.session_state.algorithm
            if 'available_params' not in st.session_state:
                st.session_state.available_params = list(parameters.keys())
            
            param_col1, param_col2 = st.columns([3, 1])
            with param_col1:
                available_params = [p for p in parameters.keys() if p not in st.session_state.manual_params]
                
                if available_params:
                    selected_param = st.selectbox("Select parameter to add:", available_params)
                else:
                    st.write("All parameters have been added.")
                    selected_param = None
            
            with param_col2:
                st.write("") 
                if selected_param and st.button("Add Parameter", key="add_param"):
                    param_type = parameters[selected_param][-1]
                    if param_type == "category":
                        st.session_state.manual_params[selected_param] = {
                            "value": parameters[selected_param][0][0],
                            "type": param_type,
                            "options": parameters[selected_param][0]
                        }
                    else: 
                        default_value = parameters[selected_param][0]
                        st.session_state.manual_params[selected_param] = {
                            "value": default_value,
                            "min": parameters[selected_param][0],
                            "max": parameters[selected_param][1],
                            "step": parameters[selected_param][2],
                            "type": param_type
                        }
            
            if st.session_state.manual_params:
                st.write("### Selected Parameters")
                
                params_to_delete = []
                
                for param in st.session_state.manual_params:
                    param_data = st.session_state.manual_params[param]
                    param_type = param_data["type"]
                    
                    param_container = st.container()
                    param_col1, param_col2 = param_container.columns([5, 1])
                    
                    with param_col1:
                        if param_type == "category":
                            selected_option = st.selectbox(
                                f"{param}:", 
                                options=param_data["options"],
                                index=param_data["options"].index(param_data["value"]) if param_data["value"] in param_data["options"] else 0
                            )
                            st.session_state.manual_params[param]["value"] = selected_option
                        
                        elif param_type == "int":
                            int_value = st.slider(
                                f"{param}:", 
                                min_value=param_data["min"],
                                max_value=param_data["max"],
                                value=param_data["value"],
                                step=param_data["step"]
                            )
                            st.session_state.manual_params[param]["value"] = int_value
                        
                        elif param_type == "float":
                            float_value = st.slider(
                                f"{param}:", 
                                min_value=float(param_data["min"]),
                                max_value=float(param_data["max"]),
                                value=float(param_data["value"]),
                                step=float(param_data["step"])
                            )
                            st.session_state.manual_params[param]["value"] = float_value
                            
                            custom_step = st.checkbox(f"Custom step size for {param}", value=False)
                            if custom_step:
                                step_value = st.number_input(
                                    f"Step size for {param}:",
                                    min_value=float(param_data["min"])/1000,
                                    max_value=float(param_data["max"])/10,
                                    value=float(param_data["step"]),
                                    format="%.6f"
                                )
                                st.session_state.manual_params[param]["step"] = step_value
                    
                    with param_col2:
                        st.write("") 
                        if st.button("Remove", key=f"remove_{param}"):
                            params_to_delete.append(param)
                    
                    st.markdown("---")
                
                for param in params_to_delete:
                    del st.session_state.manual_params[param]
                
                if st.session_state.manual_params:
                    st.write("### Parameter Summary")
                    param_summary = {param: st.session_state.manual_params[param]["value"] for param in st.session_state.manual_params}
                    st.json(param_summary)
                    
                    if st.button("Apply Parameters"):
                        st.session_state.final_params = param_summary
                        st.success("Parameters saved successfully!")
            else:
                st.info("No parameters selected. Add parameters using the selector above.")
        st.write("")
        if st.button("Continue"):
                st.session_state.page = "Summary"
                st.rerun()


# Page: Summary
elif st.session_state.page == "Summary":
    st.header("Configuration Summary")
    
    if st.session_state.df is None:
        st.warning("No data has been uploaded. Please go back to the 'Upload Data' section.")
    else:
        summary_tab1, summary_tab2, summary_tab3 = st.tabs(["Data & Features", "Model Configuration", "Preprocessing Steps"])
        
        with summary_tab1:
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(st.session_state.df))
            with col2:
                st.metric("Columns", len(st.session_state.df.columns))
            with col3:
                numeric_cols = len(st.session_state.df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            
            st.subheader("Selected Features")
            if st.session_state.feature_columns:
                st.write(f"**Feature Columns:** {', '.join(st.session_state.feature_columns)}")
            else:
                st.warning("No feature columns selected.")
                
            st.write(f"**Label Column:** {st.session_state.label_column if st.session_state.label_column else 'Not selected'}")
            
            if st.session_state.datetime_column and st.session_state.datetime_column != "None":
                st.write(f"**Datetime Column:** {st.session_state.datetime_column}")
                try:
                    min_date = st.session_state.df[st.session_state.datetime_column].min()
                    max_date = st.session_state.df[st.session_state.datetime_column].max()
                    st.write(f"**Date Range:** {min_date.strftime('%Y-%m-%d %H:%M:%S')} to {max_date.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    pass
            else:
                st.info("No datetime column selected.")
                
            if st.session_state.partition_columns:
                st.write(f"**Partition/GroupBy Columns:** {', '.join(st.session_state.partition_columns)}")
        
        with summary_tab2:
            st.subheader("Model Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**ML Task:** {st.session_state.ml_task}")
            with col2:
                st.info(f"**Algorithm:** {st.session_state.algorithm}")
            
            st.subheader("Hyperparameter Tuning")
            st.write(f"**Tuning Method:** {st.session_state.tune}")
            
            if st.session_state.tune == "Manual" and hasattr(st.session_state, 'manual_params') and st.session_state.manual_params:
                st.write("**Manual Parameters:**")
                params_df = pd.DataFrame({
                    "Parameter": list(st.session_state.manual_params.keys()),
                    "Value": [param["value"] for param in st.session_state.manual_params.values()]
                })
                st.dataframe(params_df)
            
            st.subheader("Train-Test Split")
            st.write(f"**Split Method:** {st.session_state.split_method}")
            
            if st.session_state.split_method == "Percentage Split":
                st.write(f"**Train-Test Ratio:** {st.session_state.train_test_split}")
            else: 
                try:
                    st.write(f"**Training Period:** {st.session_state.train_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.train_end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"**Testing Period:** {st.session_state.test_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.test_end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if st.session_state.ml_task == "Time Series Forecasting":
                        st.subheader("Forecast Configuration")
                        st.write(f"**Forecast Method:** {st.session_state.forecast_method}")
                        
                        if st.session_state.forecast_method == "Date Range":
                            st.write(f"**Forecast Period:** {st.session_state.forecast_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {st.session_state.forecast_end_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        else: 
                            st.write(f"**Forecast Periods:** {st.session_state.forecast_periods}")
                except:
                    st.warning("Date range information is incomplete.")
            
        with summary_tab3:
            st.subheader("Preprocessing Steps")
            if st.session_state.preprocessing_steps:
                for i, step in enumerate(st.session_state.preprocessing_steps, 1):
                    st.write(f"{i}. {step}")
            else:
                st.info("No preprocessing steps were applied.")
        
        st.subheader("Data Preview (After Preprocessing)")
        st.dataframe(st.session_state.df.head())
        
        st.subheader("Save Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Configuration"):
                save_configuration()
                
        with col2:
            config_dict = {
                "ml_task": st.session_state.ml_task,
                "algorithm": st.session_state.algorithm,
                "feature_columns": st.session_state.feature_columns,
                "label_column": st.session_state.label_column,
                "datetime_column": st.session_state.datetime_column,
                "partition_columns": st.session_state.partition_columns,
                "preprocessing_steps": st.session_state.preprocessing_steps,
                "train_test_split": st.session_state.train_test_split,
                "split_method": st.session_state.split_method,
                "hyperparameter_tuning": st.session_state.tune,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
            if st.session_state.tune == "Manual":
                config_dict.update({
                    "manual_params": st.session_state.manual_params if hasattr(st.session_state, 'manual_params') else {}
                })
            
            if st.session_state.split_method == "Time-based Split":
                try:
                    config_dict.update({
                        "train_start_date": st.session_state.train_start_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "train_end_date": st.session_state.train_end_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "test_start_date": st.session_state.test_start_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "test_end_date": st.session_state.test_end_date.strftime("%Y-%m-%d %H:%M:%S")
                    })
                except:
                    pass
            
            if st.session_state.ml_task == "Time Series Forecasting":
                try:
                    config_dict.update({
                        "forecast_method": st.session_state.forecast_method,
                        "forecast_periods": st.session_state.forecast_periods,
                        "forecast_period_unit": st.session_state.forecast_period_unit
                    })
                    
                    if st.session_state.forecast_method == "Date Range":
                        config_dict.update({
                            "forecast_start_date": st.session_state.forecast_start_date.strftime("%Y-%m-%d %H:%M:%S"),
                            "forecast_end_date": st.session_state.forecast_end_date.strftime("%Y-%m-%d %H:%M:%S")
                        })
                except:
                    pass
            
            config_str = json.dumps(config_dict, indent=2)
            
            st.download_button(
                label="Download Configuration",
                data=config_str,
                file_name=f"automl_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Back to Model Configuration"):
                st.session_state.page = "Model Configuration"
                st.rerun()
        
        with col3:
            if st.button("📊 Run AutoML Training"):
                start_time = datetime.datetime.now()
                config_dict["timestamp"] = start_time
                config_dict["model_id"] = f"model_{start_time.strftime('%Y%m%dT%H%M%S')}"
                config_dict["model_name"] = f"MN_DevTest"
                os.makedirs("Output", exist_ok=True)
                progress_file = os.path.join("Output", "progress.csv")
                if not os.path.exists(progress_file):
                    pd.DataFrame(columns=[
                        "Model_ID", "Model_Name", "ML_Task", "Algorithm", 
                        "Start_Time", "End_Time", "Execution_Time", 
                        "Progress", "Status", "Metric_Name", "Metric_Value"
                    ]).to_csv(progress_file, index=False)
                new_progress_dict = {
                    "Model_ID": config_dict["model_id"],
                    "Model_Name": config_dict["model_name"],
                    "ML_Task": config_dict["ml_task"],
                    "Algorithm": config_dict["algorithm"],
                    "Start_Time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "End_Time": "",
                    "Execution_Time": "",
                    "Progress": "0%",
                    "Status": "Running",
                    "Remarks": "",
                    "Metric_Name": "",
                    "Metric_Value": ""
                }
                new_progress_df = pd.concat([pd.read_csv(progress_file), pd.DataFrame([new_progress_dict])], ignore_index=True)
                new_progress_df.to_csv(progress_file, index=False)
                
                def run_model_training_worker(configurationJSON, exog_df):
                    try:
                        handler.start_model_training(configurationJSON, exog_df)
                        gc.collect()
                    except Exception as e:
                        end_time = datetime.datetime.now()
                        read_progress_file = os.path.join("Output", "progress.csv")
                        if os.path.exists(read_progress_file):
                            read_progress_df = pd.read_csv(read_progress_file)
                            model_idx = read_progress_df[read_progress_df["Model_ID"] == configurationJSON["model_id"]].index
                            if not model_idx.empty:
                                read_progress_df.at[model_idx[0], "Status"] = "Failed"
                                read_progress_df.at[model_idx[0], "End_Time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
                                read_progress_df.at[model_idx[0], "Execution_Time"] = str(end_time - configurationJSON["timestamp"])[:-3]
                                read_progress_df.to_csv(read_progress_file, index=False)
                thread = threading.Thread(
                    target = run_model_training_worker,
                    args = (config_dict.copy(), st.session_state.df.copy())
                )
                thread.daemon = True
                thread.start()
                
                st.success("Model added for Training.")
                clear_sessionState()
                st.session_state.page = "Progress History"
                time.sleep(0.5)
                st.rerun()


# Page: Progress History
elif st.session_state.page == "Progress History":
    st.header("Model Training Progress")
    
    def load_progress_data():
        output_dir = "Output"
        progress_file = os.path.join(output_dir, "progress.csv")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not os.path.exists(progress_file):
            progress_df = pd.DataFrame(columns=[
                "Model_ID", "Model_Name", "ML_Task", "Algorithm", 
                "Start_Time", "End_Time", "Execution_Time", 
                "Progress", "Status", "Metric_Name", "Metric_Value"
            ])
            progress_df.to_csv(progress_file, index=False)
            return progress_df
        else:
            return pd.read_csv(progress_file)
    
    progress_df = load_progress_data()
    
    tab1, tab2 = st.tabs(["Active Models", "Completed Models"])
    
    with tab1:
        active_models = progress_df[(progress_df['Progress'] != '100.00%') & (progress_df['Status'] != 'Failed')]
        
        if not active_models.empty:
            st.write(f"**{len(active_models)} Active Model(s)**")
            
            for _, model in active_models.iterrows():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        progress_text = model['Progress']
                        progress_value = float(progress_text.strip('%')) / 100 if isinstance(progress_text, str) else 0
                        
                        st.markdown(f"**{model['Model_Name']}** ({model['ML_Task']} - {model['Algorithm']})")
                        st.progress(progress_value)
                        st.caption(f"Started: {model['Start_Time']} • Progress: {model['Progress']}")
                    
                    with col2:
                        st.write("")  # Spacing
                        if st.button("Cancel", key=f"cancel_{model['Model_ID']}"):
                            st.warning(f"Cancelling model training for {model['Model_Name']}...")
                    
                    st.markdown("---")
        else:
            st.info("No active models currently training.")
    
    with tab2:
        completed_models = progress_df[(progress_df['Progress'] == '100.00%') | (progress_df['Status'] == 'Failed')]
        
        if not completed_models.empty:
            completed_models = completed_models.sort_values(by='End_Time', ascending=False)
            
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                task_filter = st.multiselect(
                    "Filter by ML Task",
                    options=sorted(completed_models['ML_Task'].unique()),
                    default=[]
                )
            
            with filter_col2:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=sorted(completed_models['Status'].unique()),
                    default=[]
                )
            
            if task_filter:
                completed_models = completed_models[completed_models['ML_Task'].isin(task_filter)]
            if status_filter:
                completed_models = completed_models[completed_models['Status'].isin(status_filter)]
            
            st.write(f"**{len(completed_models)} Completed Model(s)**")
            
            for _, model in completed_models.iterrows():
                with st.expander(f"{model['Model_Name']} - {model['ML_Task']} ({model['End_Time']})"):
                    if 'selected_output' in st.session_state: del st.session_state.selected_output
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**Training Details**")
                        st.write(f"• Algorithm: {model['Algorithm']}")
                        st.write(f"• Start Time: {model['Start_Time']}")
                        st.write(f"• End Time: {model['End_Time']}")
                        st.write(f"• Duration: {model['Execution_Time']}")
                    
                    with col2:
                        st.markdown("**Performance**")
                        if model['Status'] == "Successfully Completed": st.markdown(f"• Status: <span style='color:green'>{model['Status']}</span>", unsafe_allow_html=True)
                        elif model['Status'] == "Failed":   st.markdown(f"• Status: <span style='color:red'>{model['Status']}</span>", unsafe_allow_html=True)
                        elif model['Status'] == "Partially Completed": st.markdown(f"• Status: <span style='color:orange'>{model['Status']}</span>", unsafe_allow_html=True)
                        elif model['Status'] == "Running": st.markdown(f"• Status: <span style='color:blue'>{model['Status']}</span>", unsafe_allow_html=True)
                        
                        if "Completed" in model['Status']:
                            if isinstance(model["Remarks"], str):
                                st.write(f"• Remarks: {model['Remarks']}")
                            metric_name = model['Metric_Name'] if not pd.isna(model['Metric_Name']) else "Score"
                            metric_value = model['Metric_Value'] if not pd.isna(model['Metric_Value']) else "N/A"
                            st.write(f"• {metric_name}: {metric_value}")
                    
                    with col3:
                        st.markdown("**Actions**")
                        output_file = os.path.join("Output", f"{model['Model_ID']}.csv")
                        
                        if os.path.exists(output_file):
                            output_sample = pd.read_csv(output_file).head(5)
                            
                            with open(output_file, 'rb') as file:
                                st.download_button(
                                    label="Download",
                                    data=file,
                                    file_name=f"{model['Model_Name']}_output.csv",
                                    mime="text/csv",
                                    key=f"download_{model['Model_ID']}"
                                )
                            
                            if st.button("View", key=f"view_{model['Model_ID']}"):
                                st.session_state.selected_output = output_file
                                st.session_state.selected_model_name = model['Model_Name']
                        else:
                            st.write("Output not available")
                
                    if 'selected_output' in st.session_state and st.session_state.selected_output:
                        st.subheader(f"Output Preview: {st.session_state.selected_model_name}")
                        
                        try:
                            output_df = pd.read_csv(st.session_state.selected_output)
                            st.dataframe(output_df)
                            
                            if 'selected_ml_task' in st.session_state:
                                if st.session_state.selected_ml_task == "Time-Series Forecasting":
                                    st.subheader("Forecast Visualization")
                            
                            if st.button("Close Preview"):
                                del st.session_state.selected_output
                                del st.session_state.selected_model_name
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading output file: {e}")
        else:
            st.info("No completed models found.")
    
    st.write("---")
    refresh_col1, refresh_col2 = st.columns([4, 1])
    with refresh_col2:
        if st.button("🔄 Refresh"):
            st.rerun()

            