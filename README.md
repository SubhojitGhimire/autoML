# AutoML Playground

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive web application for automated machine learning, built with Streamlit. This tool allows users to upload their data, perform feature engineering, and train models for various ML tasks, including Time Series Forecasting, Regression, Classification, and Outlier Detection. It leverages Ray for distributed, parallel training to accelerate the model building process.

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/7336e17a-8414-4588-b545-d0f6cbb20845" />

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

<img width="1920" height="1385" alt="image" src="https://github.com/user-attachments/assets/39293c6a-e8c0-4099-b782-c5e5acd07592" />
<br>
<img width="1920" height="2030" alt="image" src="https://github.com/user-attachments/assets/430763a7-8b21-46f8-a2c0-5cec4584476d" />
<br>
<img width="1920" height="2030" alt="image" src="https://github.com/user-attachments/assets/ab2e6026-2e5f-48e6-81ff-96b777be6150" />
<br>
<img width="1920" height="1385" alt="image" src="https://github.com/user-attachments/assets/68f66f8b-28ea-4a82-bbfe-0890be16df1c" />
<br>
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/875b8a04-87b6-42a0-bc07-61749e029edd" />
<br>
<img width="1920" height="1385" alt="image" src="https://github.com/user-attachments/assets/8f9ab047-2581-4745-a756-4b5f32173ae0" />

<h1></h1>

**This README.md file has been improved for overall readability (grammar, sentence structure, and organization) using AI tools.*