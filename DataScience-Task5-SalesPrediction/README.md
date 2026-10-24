# Advertising Sales Prediction

A machine learning project that predicts product sales based on advertising budgets across TV, radio, and newspaper channels.

## Project Overview

Advertising is an important part of business strategy. This project uses historical advertising data to understand the relationship between advertising expenditure and sales and to predict sales using machine learning.

The project includes exploratory data analysis (EDA), data visualization, model training, evaluation, and an interactive web application built with Streamlit.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Dataset

The dataset contains advertising expenditure for three channels:

- **TV:** TV advertising budget
- **Radio:** Radio advertising budget
- **Newspaper:** Newspaper advertising budget

The target variable is `sales`.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Perform exploratory data analysis and visualization.
3. Split the data into training and testing sets.
4. Train and compare machine learning models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
5. Evaluate model performance using MAE, RMSE, and R².
6. Save the trained model pipeline.
7. Build an interactive sales prediction application with Streamlit.

## Features

- Exploratory data analysis and visualizations
- Comparison of multiple regression models
- Model evaluation metrics
- Random Forest feature importance visualization
- Interactive sales prediction using advertising budgets

## Project Structure

```text
DataScience-Task5-SalesPrediction/
├── sales_prediction.ipynb
├── app.py
├── Advertising (1).csv
├── sales_prediction_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sachitaaa/Sales-Prediction-Using-Python.git
cd Sales-Prediction-Using-Python
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily.
- **R² Score:** Measures how well the model explains variation in the target values.

The actual evaluation results are available in the notebook and the Streamlit application.

## Limitations

- Predictions depend on patterns in the historical dataset.
- The model uses only TV, radio, and newspaper advertising expenditure.
- Predictions are estimates and may not reflect real-world sales under different conditions.

## Future Improvements

- Deploy the Streamlit application online.
- Improve the interface and add interactive visualizations.
- Explore additional features and regression models.
- Monitor model performance on new data.

## Author

**Sachita Aryal**

Computer Engineering Student

---

*This project was developed for learning and practicing data analysis, regression modeling, and machine learning deployment.*
