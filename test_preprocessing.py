import pandas as pd
from train import preprocess_data
from train import train_model
from sklearn.metrics import accuracy_score
from train import load_data, preprocess_data

def test_missing_values_handled():
    df = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [22, None, 30],
        'SibSp': [1, 0, 0],
        'Parch': [0, 0, 0],
        'Fare': [7.25, 71.2833, 8.05],
        'Embarked': ['S', None, 'Q'],  # Make sure Q and S exist here
        'Survived': [0, 1, 0]
    })
    X, y, scaler = preprocess_data(df)
    assert not pd.isnull(df['Age']).any()
    assert not pd.isnull(df['Embarked']).any()


def test_preprocess_output_shape():
    df = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'female'],
        'Age': [22, 38, 26],
        'SibSp': [1, 1, 0],
        'Parch': [0, 0, 0],
        'Fare': [7.25, 71.2833, 7.925],
        'Embarked': ['S', 'C', 'Q'],
        'Survived': [0, 1, 1]
    })
    X, y, scaler = preprocess_data(df)
    assert X.shape[0] == len(df)
    # Number of features should be 8 as per preprocessing features list
    assert X.shape[1] == 8



def test_model_performance():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    model = train_model(X, y)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    assert accuracy >= 0.79, f"Accuracy is too low: {accuracy}"
