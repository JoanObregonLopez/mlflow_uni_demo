README
Overview
This repository contains two case studies implemented as Jupyter notebooks:

Case Classification: A CNN-based image classification of handwritten digits from the MNIST dataset.
Case Regression: A Random Forest regression model predicting California housing prices.
Both projects utilize MLflow for experiment tracking and SHAP for model interpretability.

Repository Structure
css
Copiar código
├── .gitignore
├── LICENSE
├── requirements.txt
├── src
    ├── classification
    │   └── notebooks
    │       └── case_classification.ipynb
    └── regression
        └── notebooks
            └── case_regression.ipynb
.gitignore: Specifies intentionally untracked files to ignore.
LICENSE: Contains the license information.
requirements.txt: Lists all the Python packages required to run the notebooks.
src/: Contains all source code organized by project type.
classification/: Contains notebooks and related files for classification tasks.
case_classification.ipynb: Jupyter notebook for the classification case study.
regression/: Contains notebooks and related files for regression tasks.
case_regression.ipynb: Jupyter notebook for the regression case study.
Setup and Installation
Clone the Repository
bash
Copiar código
git clone <repository-url>
Navigate to the Repository
bash
Copiar código
cd <repository-directory>
Create a Virtual Environment (Optional but Recommended)
bash
Copiar código
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies
bash
Copiar código
pip install -r requirements.txt
Case Studies
1. Case Classification: MNIST Digit Classification
Location
src/classification/notebooks/case_classification.ipynb

Description
This notebook builds a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The entire training process, including parameters and metrics, is tracked using MLflow. Model interpretability is provided using SHAP.

Key Features
Data Loading and Preprocessing: Loads the MNIST dataset and normalizes pixel values.
Model Construction: Builds a CNN architecture suitable for image classification.
Training with MLflow: Logs hyperparameters, metrics, and the trained model.
Model Evaluation: Evaluates performance on the test dataset.
SHAP Explanations: Generates SHAP values to explain model predictions.
How to Run
Navigate to the Notebook Directory

bash
Copiar código
cd src/classification/notebooks
Launch Jupyter Notebook

bash
Copiar código
jupyter notebook case_classification.ipynb
Run the Notebook

Execute all cells sequentially. Ensure that MLflow is running if you want to track the experiment.

Results
Metrics: Accuracy and loss logged in MLflow.
Model Artifacts: Saved in MLflow, including the trained model.
SHAP Images: Saved in the notebook directory and logged to MLflow.
2. Case Regression: California Housing Price Prediction
Location
src/regression/notebooks/case_regression.ipynb

Description
This notebook builds a Random Forest Regressor to predict housing prices using the California Housing dataset. It includes a full machine learning pipeline with preprocessing steps. Experiment tracking and model logging are handled by MLflow, and SHAP is used for model interpretability.

Key Features
Data Loading and Preprocessing: Handles missing values and scales features.
Pipeline Construction: Combines preprocessing and model into a scikit-learn Pipeline.
Training with MLflow: Logs parameters, metrics, and the model.
Model Evaluation: Calculates MSE, MAE, and R² score.
Feature Importance and SHAP: Analyzes feature importances and generates SHAP plots.
How to Run
Navigate to the Notebook Directory

bash
Copiar código
cd src/regression/notebooks
Launch Jupyter Notebook

bash
Copiar código
jupyter notebook case_regression.ipynb
Run the Notebook

Execute all cells sequentially. Make sure MLflow is running to track the experiment.

Results
Metrics: MSE, MAE, and R² score logged in MLflow.
Model Artifacts: Saved in MLflow, including the trained pipeline.
Feature Importances: Plots and data saved and logged.
SHAP Plots: Summary and dependence plots saved and logged.
Requirements
The requirements.txt file in the root directory lists all the dependencies required to run both notebooks. Key packages include:

mlflow
tensorflow
keras
scikit-learn
pandas
numpy
matplotlib
shap
jupyter
Install all dependencies using:

bash
Copiar código
pip install -r requirements.txt
MLflow Tracking
Both notebooks are set up to use MLflow for experiment tracking. Before running the notebooks, ensure that MLflow is installed and that the tracking server is running if you're using a remote server.

Start MLflow Tracking UI (if needed):

bash
Copiar código
mlflow ui
Notes
Data Download: The datasets are downloaded automatically using scikit-learn's data fetching utilities.
Customizing Parameters: You can modify hyperparameters directly in the notebooks before running them.
SHAP Computation: Calculating SHAP values can be computationally intensive. In the regression case study, a sample is used for efficiency.
Licensing: Review the LICENSE file for information on the usage of this repository.
Contact
For any questions or issues, please open an issue on the repository or reach out to the project maintainer.

