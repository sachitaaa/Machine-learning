# OIBSIP

# Iris Flower Classification

**Oasis Infobyte Internship | Data Science Track | Task 1**

## Objective
Train machine learning models to classify iris flowers into three species
(Setosa, Versicolor, Virginica) from their physical measurements.

## Dataset
The Iris dataset is built into scikit-learn (`sklearn.datasets.load_iris()`).
- 150 samples, 4 features: sepal length, sepal width, petal length, petal width (cm)
- 3 balanced classes (50 samples each)
- No missing values

## Approach
1. Loaded the data into a pandas DataFrame and added a readable `species` column
2. Exploratory data analysis: shape, data types, null check, descriptive statistics
3. Visualisations: pairplot and box plots for every feature, grouped by species
4. Feature discussion: which features separate the species best
5. Train/test split: 80/20 (120 train, 30 test) with `random_state=42` and `stratify=y`
6. Trained two classifiers: Logistic Regression and Random Forest
7. Evaluated both with accuracy, confusion matrix, and classification report
8. Compared the models with 5-fold cross-validation for a more reliable result

## Key Findings
- **Petal length and petal width** are the most discriminative features. Setosa is
  completely separated by them, and versicolor and virginica overlap only slightly.
- **Sepal width** is the weakest feature, with heavy overlap between species.
- All model errors were between **versicolor and virginica**, which matches the
  overlap seen in the box plots.

## Results

| Model | Test accuracy (80/20 split) | 5-fold CV accuracy |
|---|---|---|
| Logistic Regression | 0.97 (29/30) | 0.973 |
| Random Forest | 0.90 (27/30) | 0.967 |

**Best model: Logistic Regression.** It scored highest in both evaluations and is
simpler and faster. The test set has only 30 samples, so one flower changes
accuracy by 3.3%. Cross-validation shows the two models perform almost equally well.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## How to Run
```bash
git clone https://github.com/sachitaaa/OIBSIP.git
cd OIBSIP/DataScience-Task1-IrisClassification
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook iris_classification.ipynb

Then choose **Kernel → Restart & Run All**.

## Repository Structure
DataScience-Task1-IrisClassification/
├── iris_classification.ipynb
└── README.md

## Author
Sachita Aryal
