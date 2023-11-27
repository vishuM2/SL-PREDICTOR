import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def sea_level_predictor(file_path):
    # Load the data into a Pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame
    print("Sample of the data:")
    print(df.head())

    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    # Linear regression model
    X = np.array(df['Year']).reshape((-1, 1))
    y = np.array(df['CSIRO Adjusted Sea Level'])

    model = LinearRegression()
    model.fit(X, y)

    # Predictions for future years
    future_years = np.arange(1880, 2051).reshape((-1, 1))
    future_predictions = model.predict(future_years)

    # Plotting the sea level data and predictions
    plt.figure(figsize=(14, 6))
    plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], label='Observed Sea Level')
    plt.plot(future_years, future_predictions, label='Linear Regression Prediction', color='red')
    plt.title('CSIRO Adjusted Sea Level Over Time')
    plt.xlabel('Year')
    plt.ylabel('Sea Level (inches)')
    plt.legend()
    plt.show()

    # Predicted sea level for the year 2050
    prediction_2050 = model.predict([[2050]])
    print("\nPredicted Sea Level for the Year 2050:", prediction_2050[0], "inches")

if __name__ == "__main__":
    # Provide the path to your sea level data CSV file
    file_path = "path/to/your/sea_level_data.csv"
    sea_level_predictor(file_path)
