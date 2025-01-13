# Stroke Prediction Using Machine Learning

## Project Overview
This repository contains the implementation of a random forest model to predict stroke occurrences based on patient health data. The project uses a Kaggle dataset with various patient attributes (like `bmi` or `smoking status`). The main challenges were that some data was missing and that the dataset was imbalanced.

This project was part of a [challenge]{https://www.kaggle.com/competitions/ida-ml-1-challenge-summer23/leaderboard}, where my model achieved second place. It served as the final project for the course IDA-ML 1 (Intelligent Data Analysis and Machine Learning 1) at the University of Potsdam during the Summer Semester 2023.

---

## Key Features
1. **Data Preprocessing**:
   - Missing values in `bmi` and `avg_glucose_level` were replaced with the **median** of their respective columns to handle outliers and high variance.
   - Missing values in `smoking_status` (encoded as "Unknown") were replaced with `0` after converting categorical values to numerical.

2. **Handling Imbalanced Data**:
   - The dataset was highly imbalanced with far fewer positive instances (stroke cases).
   - To address this, the minority class was **upsampled** to achieve a 50/50 distribution.

3. **Model Choice**:
   - A **Random Forest Classifier** was chosen due to its high accuracy, robustness against overfitting, ease of implementation, and explainability.
   - Feature importance was extracted to identify the most significant factors influencing stroke predictions.

4. **Hyperparameter Tuning**:
   - The `GridSearchCV` method from `sklearn` was used for hyperparameter optimization.
   - 5-fold cross-validation was applied to find the best combination of parameters.

5. **Evaluation**:
   - The model achieved an **AUC ROC score of 0.85** on the evaluation set.
   - The **ROC Curve** and **Feature Importance** were visualized for better interpretability.

---

## Repository Structure

├── data/
│   └── train.csv                   
├── preprocessing.py                # Preprocessing functions for data cleaning and transformation
├── stroke_prediction.py            # Main script containing the implementation
├── README.md                       