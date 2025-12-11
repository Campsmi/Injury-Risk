import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    df = pd.read_csv(path)
    return df


def prepare_feature(df, target_col):
    if "Athlete_ID" in df.columns:
        df = df.drop(columns=["Athlete_ID"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y

def create_splits(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test