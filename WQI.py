import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import warnings


dataset = pd.read_csv('dataset1.csv')


X = dataset.iloc[:, :-1].values #input columns 

y = dataset.iloc[:, -1].values #output columns WQI


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


models = [
    ("Multiple Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("Random Forest", RandomForestRegressor()),
    ("Support Vector Machines", SVR()),
    ("Neural Network (MLP)", MLPRegressor(max_iter=500, tol=1e-6))
]


results = []
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, mse, r2))


for name, mse, r2 in results:
    print(f"{name}:")
    print("  MSE:", mse)
    print("  R2:", r2)
    print()


best_model_mse = min(results, key=lambda x: x[1])
print("Best Model (MSE):", best_model_mse[0])


best_model_r2 = max(results, key=lambda x: x[2])
print("Best Model (R-squared):", best_model_r2[0])


fig, ax = plt.subplots(figsize=(10, 6))


mse_values = [result[1] for result in results]
ax.plot(mse_values, marker='o', label='Mean Squared Error (MSE)')


r2_values = [result[2] for result in results]
ax.plot(r2_values, marker='o', label='R-squared')

ax.set_xticks(range(len(models)))
ax.set_xticklabels([result[0] for result in results], rotation=45)
ax.set_xlabel('Models')
ax.set_ylabel('Performance')
ax.set_title('Comparison of Model Performance')
ax.legend()


ax.set_xlim([-0.5, len(models) - 0.5])


plt.subplots_adjust(bottom=0.15)

plt.show()
