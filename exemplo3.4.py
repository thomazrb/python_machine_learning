from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('dados.csv')

X = data[['Dia', 'Hora', 'Assunto']]
y = data['Views']

X = pd.get_dummies(X)

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_treino, y_treino)
print(f'Score: {model.score(X_teste, y_teste):.2f}')

previsoes = model.predict(X_teste)

mse = mean_squared_error(y_teste, previsoes)
r2 = r2_score(y_teste, previsoes)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')