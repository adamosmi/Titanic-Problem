# Titanic Survival Prediction Project

## Introduction

This project aims to predict whether a passenger on the Titanic would have survived or not, based on features like passenger class, sex, age, etc. This is a binary classification problem, and is a popular beginner's project in the field of data science and machine learning. The dataset for this project is taken from the Kaggle competition - Titanic: Machine Learning from Disaster.

## Dataset

The dataset consists of the following features:

- `PassengerId`: An unique identifier for the passenger.
- `Survived`: Whether the passenger survived or not. (0 = No; 1 = Yes)
- `Pclass`: Passenger class (1 = 1st; 2 = 2nd; 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Sex of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

The dataset has been split into two parts:

1. `train.csv`: This dataset is used for training the model. It includes the `Survived` feature.
2. `test.csv`: This dataset is used for testing the model. It does not include the `Survived` feature.

## Methodology

The project involves the following steps:

1. Exploratory Data Analysis (EDA)
2. Data Cleaning and Preprocessing
3. Model Development and Tuning
4. Model Evaluation
5. Model Deployment

## Results

The model achieves an accuracy of XX% on the test set.

## Requirements

The project requires Python 3.7+ and the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- joblib

Install these libraries with pip:

pip install -r requirements.txt

## Usage

See the `README.md` file for usage instructions.

## License

This project uses the [MIT License](LICENSE.md).

## Contact

For any queries, please contact:

- Name: Andrew Damon-Smith
- Email: andrewdamonsmith@gmail.com

