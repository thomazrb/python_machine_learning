from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dia = input('Digite o Dia: ')
hora = input('Digite a Hora: ')
assunto = input('Digite o Assunto: ')
novo_registro = pd.DataFrame({'Dia': [dia], 'Hora': [hora], 'Assunto': [assunto]})

data = pd.read_csv('dados.csv')

X = data[['Dia', 'Hora', 'Assunto']]
y = data['Views']

X = pd.get_dummies(X, columns=['Dia', 'Assunto'])
novo_registro = pd.get_dummies(novo_registro, columns=['Dia', 'Assunto'])
novo_registro = novo_registro.reindex(columns=X.columns, fill_value=0)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)
previsoes_novos_dados = model.predict(novo_registro)
print(f'Previsão de visualizações para os novos dados: {previsoes_novos_dados}')