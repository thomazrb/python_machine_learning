from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.DataFrame({
        'Temperatura': [30, 25, 20, 15, 32, 28],
        'Umidade': [60, 70, 80, 90, 55, 75],
        'Choveu': [1, 1, 0, 0, 1, 1]
    })

X = data[['Temperatura', 'Umidade']]
y = data['Choveu']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

novo_dado = pd.DataFrame({
    'Temperatura': [30, 12, 40],
    'Umidade': [90, 85, 10]
})

previsao = model.predict(novo_dado)
print(previsao)