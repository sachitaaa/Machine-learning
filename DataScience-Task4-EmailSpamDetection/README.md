# SMS Spam Detection Using Machine Learning

## Project Overview

This project focuses on building a machine learning system that classifies SMS messages as **Spam** or **Ham (Legitimate Message)**. It was developed as part of the Oasis Infobyte Data Science Internship (OIBSIP), Task 4.

## Objectives

- Analyze and preprocess SMS text data.
- Convert text messages into numerical features using TF-IDF.
- Train and compare machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Save trained models for future use.

## Dataset

The project uses an SMS spam dataset containing legitimate and spam messages.

The data was cleaned by removing unnecessary columns, handling duplicate records, and preparing labels for binary classification.

- **Ham:** Legitimate messages, labeled `0`
- **Spam:** Unwanted or fraudulent messages, labeled `1`

## Methodology

1. **Data Cleaning:** Removed unnecessary columns and duplicate messages.
2. **Exploratory Data Analysis:** Examined class distribution and message lengths.
3. **Text Preprocessing:** Applied lowercasing, punctuation removal, stopword removal, and stemming.
4. **Feature Extraction:** Used TF-IDF vectorization to represent messages numerically.
5. **Model Training:** Trained Multinomial Naive Bayes and Logistic Regression.
6. **Evaluation:** Compared accuracy, precision, recall, F1-score, and confusion matrices.
7. **Model Saving:** Saved both trained models and the TF-IDF vectorizer using Python's pickle module.

## Models Used

- Multinomial Naive Bayes
- Logistic Regression

Naive Bayes achieved higher recall for the spam class in the evaluation performed for this project. Recall is important because it measures how many actual spam messages the model successfully identifies.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib and Seaborn
- NLTK
- Scikit-learn
- Jupyter Notebook
- Pickle

## Project Files

- `spam_detection.ipynb` — notebook containing data analysis, preprocessing, training, and evaluation.
- `naive_bayes_model.pkl` — saved Naive Bayes model.
- `logistic_regression_model.pkl` — saved Logistic Regression model.
- `tfidf_vectorizer.pkl` — saved TF-IDF vectorizer.


## How to Run

1. Clone or download this repository.
2. Install the required libraries:

   
   pip install pandas numpy matplotlib seaborn nltk scikit-learn jupyter


3. Open the notebook:
   bash jupyter notebook
   

4. Run the notebook cells in order to reproduce the analysis and training process.

## Conclusion

This project demonstrates a complete text classification workflow, from data cleaning and text preprocessing to feature extraction, model evaluation, and saving trained models. It provided practical experience in applying machine learning techniques to a real-world text classification problem.