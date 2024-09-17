# Solar Irradiance Prediction

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Modeling Approaches](#modeling-approaches)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Prediction](#prediction)
  - [Combining Models](#combining-models)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [License](#license)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Overview

This repository is dedicated to the prediction of solar irradiance using machine learning models. Solar irradiance, the power per unit area received from the Sun, is a critical factor in the performance and efficiency of solar energy systems. Accurately predicting solar irradiance helps in optimizing solar energy production, improving energy management, and enhancing grid stability.

This project is part of our final year college project, where we explore various machine learning techniques to forecast solar irradiance based on historical meteorological data. We implemented and compared several regression and classification models to identify the most accurate and reliable approaches.

## Motivation

The rapid growth in renewable energy has increased the importance of accurate solar irradiance prediction. Solar power, being a major component of the renewable energy mix, requires precise forecasting to maximize efficiency and ensure a reliable supply. This project aims to contribute to this field by developing machine learning models that can predict solar irradiance with high accuracy.

## Project Structure

The project is organized into the following directories and files:

- **`.github/workflows/`**: Contains the CI/CD pipeline configurations for the project.
- **`DataSet/`**: Includes the dataset used for model training and testing. This dataset contains historical data on solar irradiance and other relevant meteorological parameters.
- **`Results/`**: Stores the results of model predictions, performance metrics, and comparison charts.
- **Model Files**:
  - **Adaboost.ipynb**: Jupyter notebook implementing the AdaBoost Regressor model.
  - **Gradient Boosting Regressor.py**: Python script for training and evaluating the Gradient Boosting Regressor.
  - **Random Forest Regressor.py**: Python script for the Random Forest Regressor model.
  - **XGBoost.ipynb**: Jupyter notebook implementing the XGBoost model.
  - Various scripts for other models like CatBoost, Gaussian Process Regression, LightGBM, and NGBoost, both for classification and regression tasks.
- **`final results.xlsx`**: Excel file summarizing the performance metrics of all the models tested.
- **`requirements.txt`**: List of Python dependencies required to run the models.
- **`LICENSE`**: The MIT license file for the project.
- **`README.md`**: This detailed documentation of the project.

## Dataset Description

The dataset used in this project includes the following features:

- **Date and Time**: Timestamps for each recorded observation.
- **Solar Irradiance**: Measured in watts per square meter (W/m²), this is the target variable we aim to predict.
- **Temperature**: Ambient temperature data in degrees Celsius.
- **Humidity**: Relative humidity percentage.
- **Wind Speed**: Wind speed in meters per second.
- **Cloud Cover**: Percentage of sky covered by clouds, an important feature affecting solar irradiance.
- **Atmospheric Pressure**: Measured in hPa, affecting weather patterns and irradiance levels.

The dataset is preprocessed to handle missing values, normalize features, and split into training and testing subsets.

## Modeling Approaches

We employed several machine learning models to predict solar irradiance, including:

1. **AdaBoost Regressor**: A boosting technique that combines multiple weak learners to form a strong predictor.
2. **Gradient Boosting Regressor**: Another boosting algorithm that builds models sequentially, with each model correcting the errors of the previous one.
3. **Random Forest Regressor**: An ensemble of decision trees that improves prediction accuracy by averaging multiple decision trees trained on different parts of the dataset.
4. **XGBoost**: An optimized gradient boosting algorithm known for its speed and performance.
5. **CatBoost**: A gradient boosting algorithm that handles categorical features natively.
6. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms.
7. **Gaussian Process Regression**: A non-parametric, Bayesian approach to regression.
8. **NGBoost**: Natural Gradient Boosting, a probabilistic prediction algorithm.

For classification tasks, corresponding classifier versions of these models were used.

## Installation

To set up the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tejaschaudhari131/Solar-Irradiance-prediction-.git
   cd Solar-Irradiance-prediction-
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Ensure that the dataset is correctly placed in the `DataSet/` directory. The data should be cleaned and preprocessed as needed. The scripts are designed to handle the necessary preprocessing steps like normalization and splitting the data into training and testing sets.

### Model Training

To train a specific model, run the corresponding script or Jupyter notebook. For example, to train the XGBoost model, open and execute the cells in `XGBoost.ipynb`.

### Model Evaluation

After training, the models are evaluated using several performance metrics:

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

These metrics help compare the performance of different models. The results are saved in the `Results/` folder and summarized in the `final results.xlsx` file.

### Prediction

Once trained, the models can be used to make predictions on new data. The prediction scripts take input features and output the predicted solar irradiance values.

### Combining Models

To improve prediction accuracy, you can use the `solar_prediction_combined_model.py` script. This script combines the outputs of different models using ensemble techniques like averaging or voting to provide a more robust prediction.

## Results

The results of our experiments are documented in the `final results.xlsx` file, which includes:

- Comparison of model performance metrics.
- Graphs and charts visualizing the performance of each model.
- Insights into which models performed best under different conditions.

Our experiments show that ensemble models, particularly Gradient Boosting and XGBoost, provided the most accurate predictions of solar irradiance.

## Conclusion

This project demonstrates the application of various machine learning techniques to predict solar irradiance with high accuracy. The results highlight the effectiveness of ensemble methods like Gradient Boosting and XGBoost in forecasting tasks involving complex and non-linear relationships.

## Future Work

Potential future enhancements to this project include:

- **Incorporating More Features**: Adding additional meteorological variables or satellite data could improve model accuracy.
- **Real-Time Prediction**: Implementing a real-time prediction system that continuously updates based on new data.
- **Deployment**: Deploying the best-performing model as a web service or API for use by other applications.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this project as per the terms of the license. See the [LICENSE](LICENSE) file for more details.

## Contributing

As this is a final year college project, contributions are not expected. However, if you have suggestions for improvements, feel free to fork the repository and submit a pull request. We welcome constructive feedback and ideas for further development.

