# Car Price Prediction

## Project Overview
This project predicts the selling price of a used car using machine learning. It explores car features, preprocesses the data, trains regression models, evaluates their performance, and saves the selected model for future predictions.

## Dataset
The dataset contains information about used cars, including:
- Car name and manufacturing year
- Present price
- Kilometers driven
- Fuel type
- Seller type
- Transmission type
- Number of previous owners

After removing duplicate rows, the dataset contained **299 records**.

## Data Preprocessing
- Removed duplicate records.
- Created a `Car_Age` feature from the manufacturing year.
- Extracted a first-token category from `Car_Name` as `Brand`.
- Separated the features (`X`) and target (`y`).
- Applied one-hot encoding to categorical features.
- Split the data into 80% training and 20% testing sets.

## Machine Learning Models
The following regression models were evaluated:
1. Linear Regression
2. Decision Tree Regressor
3. Gradient Boosting Regressor

## Model Evaluation

| Model | MAE | MSE | RMSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 1.4741 | 7.8164 | 2.7958 | 0.6967 |
| Decision Tree Regressor | 1.0105 | 3.9969 | 1.9992 | 0.8449 |
| Gradient Boosting Regressor | 1.1336 | 7.1031 | 2.6652 | 0.7244 |

Five-fold cross-validation mean R² scores:
- Linear Regression: 0.7508
- Decision Tree Regressor: 0.8683
- Gradient Boosting Regressor: 0.8675

The Decision Tree Regressor achieved the best test-set R² score among these models.

## Model Saving and Prediction
The selected model was saved as `car_price_model.pkl` using Joblib and then loaded again to verify that it could make predictions.

A sample car input produced a predicted selling price of **7.90 lakhs**. This is an example prediction, not a guarantee of a car's actual market price.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (if used in the notebook)
- Joblib
- Jupyter Notebook

## How to Run
1. Install the required Python libraries.
2. Open `Prediction.ipynb` in Jupyter Notebook.
3. Run the notebook cells in order.


## Run the Streamlit App Locally

### 1. Clone the repository

```bash
git clone https://github.com/sachitaaa/OIBSIP.git
cd OIBSIP/DataScience-Task3-CarPricePrediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

The app will open in your browser, where you can enter car details and get a predicted selling price.


The dataset is provided as `car data.csv`, and the saved model is `car_price_model.pkl`.

## Limitations
The model's predictions depend on the dataset and selected features. The Decision Tree may overfit the training data, so predictions should be treated as estimates rather than exact market prices.

