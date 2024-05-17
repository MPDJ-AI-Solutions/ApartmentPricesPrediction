import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import winsound

def load_data(source_list: list):
    return pd.concat([pd.read_csv(data_set)for data_set in source_list])

def notify(time):
    winsound.Beep(3000, time)  # Beep at 1000 Hz for 100 ms


def evaluate_model(true_data, predicted_data):
    mse = mean_squared_error(predicted_data, true_data)
    mae = mean_absolute_error(predicted_data, true_data)
    r2 = r2_score(predicted_data, true_data)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R-squared: {r2}')


def fill_na(df, column_list, method='median'):
    for column in column_list:
        fill_value = None
        match method:
            case 'median':
                fill_value = df[column].median()
            case 'mean':
                fill_value = df[column].mean()
            case 'first_value':
                fill_value = df[column][1]
            case 'false':
                fill_value = False
            case _:
                fill_value = 0
                
        df[column] = df[column].fillna(fill_value)


def normalize_numerical_columns(df, column_list):
    df[column_list] = (df[column_list] - df[column_list].min()) / (df[column_list].max() - df[column_list].min())


def normalize_data(df: pd.DataFrame, numerical_columns: list =[], categorical_columns: list=[], boolean_columns: list=[], drop_columns: list=[], fill_method:str='mean'):
    fill_na(df, boolean_columns, 'false')
    fill_na(df, numerical_columns, fill_method)

    df = pd.get_dummies(df, columns=categorical_columns)
    df = pd.get_dummies(df, columns=boolean_columns, drop_first=True).astype(int)

    normalize_numerical_columns(df, numerical_columns)

    return df


def remove_exceptions(df: pd.DataFrame):
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df['price'] >= lower_bound)&(df['price'] <= upper_bound)]


def split_train_test(df: pd.DataFrame, output: str):
    X = df.drop(output, axis=1)
    y = df[output]
    return train_test_split(X, y, test_size=0.2, random_state=42)
