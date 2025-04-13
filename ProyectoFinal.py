import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

file_path = r'C:\Users\user\Desktop\Git\03-Instalaciones\healthcare-dataset-stroke-data.csv'

data = pd.read_csv(file_path)

data['bmi'].fillna(data['bmi'].median(), inplace=True)

data_encoded = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

X = data_encoded.drop(['id', 'stroke'], axis=1)
y = data_encoded['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred_linreg = linreg.predict(X_test)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
mse_linreg = mean_squared_error(y_test, y_pred_linreg)
r2_linreg = r2_score(y_test, y_pred_linreg)

# Mostrar los resultados
print(f"Precisión de la Regresión Logística: {accuracy_logreg:.4f}")
print(f"Error Cuadrático Medio (MSE) de la Regresión Lineal: {mse_linreg:.4f}")
print(f"R² de la Regresión Lineal: {r2_linreg:.4f}")
