# Titanic Survival Prediction Project

This repository contains all the code and analysis for the Kaggle Titanic survival prediction project.

## Project Overview

The goal of this project is to use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. We make use of the Titanic dataset provided by Kaggle and use exploratory data analysis, data cleaning, and machine learning to create this predictive model.

## Dataset

The dataset used in this project is the famous Titanic dataset from Kaggle. It contains passenger information like name, age, gender, socio-economic class, etc. The 'Survived' column is the target label: if Suvived = 1, the passenger survived, otherwise they did not.

The raw dataset files are:

- data/raw/train.csv
- data/raw/test.csv

## Method

The project uses a Random Forest classifier for prediction. Data cleaning and preprocessing is performed using pandas. The project is implemented in Python.

## Repository Structure

This repository contains the following files and folders:

1. `data/` : This folder contains raw and processed data.
2. `docs/` : This folder contains documentation of the project.
3. `notebooks/` : This folder contains Jupyter notebooks for exploratory data analysis and initial experimentation.
4. `src/` : This folder contains source code for this project.
5. `tests/` : This folder contains test code.
6. `models/` : This folder contains the trained and serialized models, model predictions, or model summaries.
7. `reports/` : This folder contains generated analysis reports.
8. `requirements.txt` : This file lists the Python dependencies required by this project.

## Installation

1. Clone this repository.
2. Run `pip install -r requirements.txt` to install the necessary dependencies.
3. (Optional) Set up a virtual environment for running the code.

## Usage

To train the model, run the script with the command: `python src/models/train_model.py`.

To make predictions using the trained model, run the script with the command: `python src/models/predict.py`.

## Authors

Andrew Damon-Smith

## License

MIT License
