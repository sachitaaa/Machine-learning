# Iris Flower Classification

## Project Overview

This project uses machine learning to classify iris flowers into three species: **Setosa, Versicolor, and Virginica**, based on their sepal and petal measurements.

It was developed as part of **Task 1 of the Oasis Infobyte Internship Program (OIBSIP)** under the Data Science track.

## Objectives

- Explore and understand the Iris dataset through Exploratory Data Analysis (EDA).
- Visualize relationships between flower measurements and species.
- Train and compare machine learning classification models.
- Evaluate model performance using appropriate classification metrics and cross-validation.

## Dataset

The project uses the Iris dataset, which contains 150 samples, four numerical features, and a target label representing the flower species.

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

- Setosa
- Versicolor
- Virginica

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Load the Iris dataset.
2. Inspect the dataset and check for missing values.
3. Perform Exploratory Data Analysis (EDA).
4. Visualize feature distributions and relationships.
5. Split the data into training and testing sets.
6. Train Logistic Regression and Random Forest classifiers.
7. Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Apply 5-fold cross-validation to assess model consistency.

## Model Evaluation

The project compares two classification algorithms:

- **Logistic Regression:** A classification algorithm that estimates class probabilities using the input features.
- **Random Forest:** An ensemble learning algorithm that combines predictions from multiple decision trees.

The models are evaluated using classification reports, confusion matrices, and 5-fold cross-validation.

Refer to the Jupyter Notebook for the actual results, visualizations, and detailed analysis.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/sachitaaa/OIBSIP.git
cd OIBSIP/DataScience-Task1-IrisClassification
```

### 2. Install the dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `iris_classification.ipynb` and run the notebook cells in order.

## Repository Structure

```text
DataScience-Task1-IrisClassification/
├── README.md
└── iris_classification.ipynb
```

## Learning Outcomes

- Practiced exploratory data analysis and data visualization.
- Learned to train and compare classification models.
- Understood confusion matrices and classification metrics.
- Explored 5-fold cross-validation for more reliable model evaluation.
- Practiced documenting a machine learning project for reproducibility.

## Author

**Sachita Aryal**

GitHub: [@sachitaaa](https://github.com/sachitaaa)

## Internship

Oasis Infobyte Internship Program (OIBSIP) — Data Science Track.
