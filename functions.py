from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import winsound

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
