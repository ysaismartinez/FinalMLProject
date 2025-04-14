import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scripts.preprocess import load_and_preprocess_data

# This script trains a random forest. 
# For this project we will be showing output for both, a random forest and a neural network
# For the sake of modularity, we split into distinct scripts

def train_random_forest():

    # First let's load and preprocess data

    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    # Let's definine a random forest regressor variable and assign a random state of 42

    rf = RandomForestRegressor(random_state=42)

    # Let's define a parameter grid

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }

    # Let's define a grid search CV variable

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print(f"Random Forest R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # Finally let's save model and preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/rf_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    print("Model and preprocessor saved to models/")

if __name__ == "__main__":
    train_random_forest()
