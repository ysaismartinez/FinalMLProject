# FinalMLProject
This is the repo for my final ML project. 
# Forest Fire Area Prediction

This project builds and compares two machine learning models to predict the burned area of forest fires using meteorological and environmental data. The dataset is from the UCI Machine Learning Repository and contains structured tabular data.

##  Dataset

- **Source**: [UCI Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/forest+fires)
- **Target**: `area` (burned area of the forest in hectares)
- **Features**: Weather indices (e.g., temperature, wind, humidity), date, and spatial coordinates

## Models

Two models were trained and evaluated:

1. **Random Forest Regressor**
   - Built using `scikit-learn`
   - Tuned using `GridSearchCV`
   - Saved as: `models/rf_model.pkl`

2. **Neural Network**
   - Built from scratch using `PyTorch`
   - Fully connected architecture
   - Trained for 100 epochs
   - Saved as: `models/nn_model.pth`

## Instructions to run this as I used python scripts not a junyper workbook. 


## 🧪 Evaluation Results (Example)

| Metric      | Random Forest | Neural Network |
|-------------|----------------|----------------|
| MSE         | 17.35          | 21.12          |
| R² Score    | 0.76           | 0.70           |

*Note: Results will vary depending on random splits and tuning.*

## 🚀 How to Run

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
2. pip install -r requirements.txt
3. python3 -m scripts.train_rf
4. python3 -m scripts.train_nn
5. python3 -m scripts.evaluate

