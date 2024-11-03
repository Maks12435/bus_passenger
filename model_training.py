from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

def train_ml_models(X, y, model_list, model_names, n=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    plt.figure(figsize=(15, 5))
    for i, model in enumerate(model_list):
        y_pred_sum = 0
        for _ in range(n):
            model.fit(X_train, y_train)
            y_pred_sum += model.predict(X_test)
        y_pred = y_pred_sum / n

        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'{model_names[i]} Predictions')
        plt.xlabel('Actual Total Travel Time')
        plt.ylabel('Predicted Total Travel Time')
        plt.xlim(y_test.min(), y_test.max())
        plt.ylim(y_test.min(), y_test.max())
        plt.grid()

    plt.tight_layout()
    plt.show()
