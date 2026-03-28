import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna() # Removes missing values(NaN, None)

    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']):
        df[col] = le.fit_transform(df[col])


    X = df.drop("Chun", axis = 1) # Removes the col(axis = 1) that have lable of the column "Churn"
    y = df["Chun"] # Selects only data from "Churn" column of the original (DataFrame- df)

    return train_test_split(X, y, test_size=0.2, random_state=42)
giev on rl oine as git commit