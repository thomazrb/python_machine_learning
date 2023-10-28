from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.DataFrame({
        'Temperatura': [30, 25, 20, 15, 32, 28, 22, 18, 26, 29],
        'Umidade': [60, 70, 80, 90, 55, 75, 78, 68, 72, 62],
        'Choveu': [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
    })

X = data[['Temperatura', 'Umidade']]
y = data['Choveu']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_treino, y_treino)

acuracia = model.score(X_teste, y_teste)
print(f'Acurácia do modelo no conjunto de teste: {acuracia:.2f}')
previsoes = model.predict(X_teste)
for i, previsao in enumerate(previsoes):
    if previsao == 1:
        resultado = 'Choveu'
    else:
        resultado = 'Não choveu'
    print(f'Amostra {i + 1}: Temperatura={X_teste.iloc[i]["Temperatura"]}, Umidade={X_teste.iloc[i]["Umidade"]} => {resultado}')