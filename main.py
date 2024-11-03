from data_processing import load_and_preprocess_data
from visualization import plot_correlation_matrix, plot_dwell_and_travel_times
from model_training import train_ml_models
from neural_network import build_and_train_nn
from predict import predict_travel_time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = load_and_preprocess_data('Bus.xlsx')
plot_correlation_matrix(data)

sample = data[data['direction'] == 1]
X = sample[['bus_stop', 'hour', 'day_of_week', 'dwell_time_in_seconds', 'travel_time']]
y = sample['total_travel_time']
models = [DecisionTreeRegressor(), KNeighborsRegressor(), RandomForestRegressor()]
model_names = ["Decision Tree", "K-Nearest Neighbors", "Random Forest"]

train_ml_models(X, y, models, model_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
nn_model = build_and_train_nn(X_train, y_train, X_test, y_test)

average_dwell_time = X['dwell_time_in_seconds'].mean()
average_travel_time = X['travel_time'].mean()
predict_travel_time(nn_model, scaler, 101, average_dwell_time, average_travel_time)


