# AutoML Playground

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive web application for automated machine learning, built with Streamlit. This tool allows users to upload their data, perform feature engineering, and train models for various ML tasks, including Time Series Forecasting, Regression, Classification, and Outlier Detection. It leverages Ray for distributed, parallel training to accelerate the model building process.

<img width="1919" height="1016" alt="Image: Upload Data" src="https://github.com/user-attachments/assets/c701971f-8c8a-40ca-8b12-063bd0330342" />

---

## Key Features
- Interactive UI: A multi-page Streamlit application that guides the user from data upload to model training.
- Multiple ML Tasks: Supports a wide range of tasks:
  - Time Series Forecasting
  - Regression
  - Classification
  - Outlier/Anomaly Detection
- Diverse Algorithm Selection: Implements popular and powerful algorithms for each task.
  <table>
      <tr align="center">
          <td><strong>Machine Learning Task</strong></td>
          <td><strong>Supported Algorithms</strong></td>
      </tr>
      <tr>
          <td>Time-Series Forecasting</td>
          <td>ARIMA, SARIMA, Neural Prophet, Attention-Based LSTM, XGBoost Forecaster</td>
      </tr>
      <tr>
          <td>Regression</td>
          <td>XGBoost Regressor, Random Forest Regressor, Linear Regression, SVM Regressor (SVR)</td>
      </tr>
      <tr>
          <td>Classification</td>
          <td>XGBoost Classifier, Logistic Regression, Linear SVM Classifier (LinearSVC)</td>
      </tr>
      <tr>
          <td>Outlier Detection</td>
          <td>CatBoost Outlier Detector, Local Outlier Factor (LOF), One-Class SVM</td>
      </tr>
  </table>
  
- Hyperparameter Tuning:
  - Auto-tuning: Utilizes Bayesian Optimization (scikit-optimize) to automatically find the best hyperparameters.
  - Manual Tuning: Provides an interface for users to manually set and experiment with algorithm parameters.
- Distributed Computing: Uses the Ray framework to distribute model training across multiple processes or nodes, perfect for large datasets or partitioned data.
- Comprehensive Preprocessing: Offers UI-based controls for:
  - Missing value imputation
  - Feature scaling (Standardization & Normalization)
  - Data sorting and randomization
  - Datetime format conversion
- Real-time Progress Tracking: A dedicated "Progress History" page to monitor active training jobs and review completed ones.
- Data Partitioning: Ability to train models in parallel on different segments of the data (e.g., for different stores or products).

# Requirements and Usage

Clone the repository. From inside the repo folder, install the dependencies:
```bash
python -m pip install --upgrade -r .\requirements.txt
```

Run the Streamlit application:
```bash
streamlit run .\main.py
```
The application should now be open and accessible in your web browser. The application is designed to be intuitive, guiding you through a series of steps in the sidebar.
1. Upload Data: Start by uploading your dataset (CSV or Excel) or providing a local file path.
2. Data Overview: Get a summary of your data, including shapes, data types, and a correlation matrix to understand feature relationships.
3. Feature Engineering:
   - Select your feature, label, and (optional) datetime and partition columns.
   - Apply preprocessing steps like handling missing values, scaling features, or sorting data.
4. Model Configuration:
   - Choose the ML task you want to perform.
   - Select an algorithm from the list of supported models.
   - Choose your hyperparameter tuning strategy ("Auto" for bayesian hyperparameter tuning or "Manual" for user-specific values).
   - Define your train/test split strategy (percentage-based or time-based).
5. Summary: Review all your configurations in one place. You can download the configuration as a JSON file for reproducibility.
6. Run AutoML Training: Click the button to start the training process. The job will be sent to the Ray backend for execution.
7. Progress History: Navigate to this page to see your model's training progress in real-time or view the results of completed jobs. You can download the prediction outputs from here.

## Screenshots

<img width="1920" height="1722" alt="Image: Data Overview" src="https://github.com/user-attachments/assets/e8684928-480e-4632-be16-68edf797639f" />
<br>
<img width="1920" height="2214" alt="Image: Feature Engineering" src="https://github.com/user-attachments/assets/96d52f3b-3279-42cf-ae48-9104d5b6d38f" />
<br>
<img width="1921" height="2285" alt="Image: Model Configuration" src="https://github.com/user-attachments/assets/432b7ba5-953f-4b52-8c99-0892003a70d5" />
<br>
<img width="1920" height="912" alt="Image: Progress History Active Model" src="https://github.com/user-attachments/assets/c2a62aa4-2a38-4ca5-84cf-83d6e7f65c39" />
<br>
<img width="1920" height="2108" alt="Image: Progress History Completed Model" src="https://github.com/user-attachments/assets/37f860a5-9397-40c5-983f-dcdad01e4009" />
<br>
<img width="1451" height="1407" alt="Image: Terminal" src="https://github.com/user-attachments/assets/f30b5f70-e895-467e-83f2-ea80c9d2c2a4" />

<h1></h1>

**This README.md file has been improved for overall readability (grammar, sentence structure, and organization) using AI tools.*
