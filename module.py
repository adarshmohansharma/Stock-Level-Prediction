import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Global Constants
K = 5  # Number of folds for cross-validation
SPLIT = 0.8  # Train-test split ratio

# Load data from CSV file
def load_data(path: str):
    """
    Load data from a CSV file into a Pandas DataFrame.

    :param path: str, relative path of the CSV file
    :return df: pd.DataFrame
    """
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(data: pd.DataFrame, target: str):
    """
    Split the columns into a target column and a set of predictor variables.

    :param data: pd.DataFrame, dataframe containing data for the model
    :param target: str, target variable that you want to predict
    :return X: pd.DataFrame, predictor variables
    :return y: pd.Series, target variable
    """
    if target not in data.columns:
        raise ValueError(f"Target variable '{target}' is not present in the data.")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Train algorithm with cross-validation
def train_algorithm_with_cross_validation(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest Regressor model using K-fold cross-validation.

    :param X: pd.DataFrame, predictor variables
    :param y: pd.Series, target variable
    """
    # Create a list to store the mean absolute errors (MAE) of each fold
    mae_scores = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(K):
        # Instantiate the algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=SPLIT, random_state=42)

        # Scale X data to help the algorithm converge and avoid greedy behavior
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Generate predictions on the test sample
        y_pred = model.predict(X_test_scaled)

        # Compute the mean absolute error for the fold
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mae_scores.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Compute the average MAE across all folds
    average_mae = sum(mae_scores) / len(mae_scores)
    print(f"Average MAE: {average_mae:.2f}")

# Main function to execute the pipeline
def main():
    # Replace 'path/to/your/data.csv' with the actual path to the CSV file containing the data
    csv_file_path = 'path/to/your/data.csv'
    target_variable = "estimated_stock_pct"  # Replace with the target variable column name

    # Load data
    data = load_data(csv_file_path)

    # Create target and predictor variables
    X, y = create_target_and_predictors(data, target_variable)

    # Train algorithm with cross-validation
    train_algorithm_with_cross_validation(X, y)

if __name__ == "__main__":
    main()
