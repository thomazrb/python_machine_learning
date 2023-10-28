from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [3, 6, 9, 12, 15]

modelo = LinearRegression() # Modelo em branco...
modelo.fit(X,y)             # Treinamos o modelo (aprender)

numero = 6
previsao = modelo.predict([[numero]])

print(previsao)
