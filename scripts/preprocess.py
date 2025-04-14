import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# The method below loads the data for processing
def load_and_preprocess_data(path=None):
    # Since absolute and relative paths can be confusing and we all think about them depending on OS
    # I created this logic using builtin functionality from the OS import
    # And it basically just checks if no path is giving and finds it in the root directory
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, '..', 'data', 'forestfires.csv')

    # This is just a quick check to make sure that we are looking into the correct path
    print("Attempting to load dataset from: "+path) 

    # Everything below is pretty standard
    # I recycled most of it from Dr Bent's class last semester
    df = pd.read_csv(path)
    df = pd.read_csv(path)

    X = df.drop(columns=['area'])
    y = df['area']

    categorical_cols = ['month', 'day']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train.values, y_test.values, preprocessor
