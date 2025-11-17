import pandas as pd
from sklearn.model_selection import train_test_split

def import_data():

    file_path = 'data/collegiate_athlete_injury_dataset.csv'
    data = pd.read_csv(file_path)

    data = data.drop(columns=["Athlete_ID"])
    data = data.dropna()

    label = 'ACL_Risk_Score'
    y = data[label]
    X = data.drop(columns=[label])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test