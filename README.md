# Sales Prediction using Python (Machine Learning)

This project predicts product sales based on advertising budgets using Linear Regression in Python.

## Project Overview
The model is trained using a dataset containing advertising budgets for TV, Radio, and Newspaper.  
It predicts the expected Sales value based on the given inputs.

## Features
- Loads dataset from CSV file
- Splits data into training and testing sets
- Trains a Linear Regression model
- Evaluates model performance using MAE and R² Score
- Saves trained model as a .pkl file
- Allows user input for sales prediction

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib

## Files in this Repository
- sales_prediction.py : Main Python code
- sales.csv : Dataset used for training
- sales_prediction_model.pkl : Saved trained model
- README.md : Project documentation

## How to Run

### Install required libraries
```bash
pip install pandas scikit-learn joblib
