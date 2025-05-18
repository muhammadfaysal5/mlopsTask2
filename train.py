import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_data():
    # You can also download from URL
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

def preprocess_data(df):
    # Fix FutureWarning by avoiding inplace chained assignment
    df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
    df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    
    X = df[features]
    y = df['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_model(X, y):
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


def save_model(model, scaler):
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

def main():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, scaler)
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
