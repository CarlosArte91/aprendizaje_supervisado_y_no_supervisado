import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from colorama import init

init(autoreset=True)

datos = pd.read_csv('ciudades_colombia.csv')

caracteristicas = datos[['temperatura_promedio', 'precipitacion_promedio']]

etiqueta = datos['tipo_de_clima']

caract_entre, caract_prue, etiq_entre, etiq_prue = train_test_split(caracteristicas, etiqueta, test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier()

modelo.fit(caract_entre, etiq_entre)

presicion = modelo.score(caract_prue, etiq_prue)
print(f'Precisión del modelo en el conjunto de prueba: {presicion * 100:.0f}%')

print('\n************ DATOS DE LA CIUDAD PARA LA PREDICCIÓN DE SU CLIMA ************\n')
ciudad = input('Ingresa el nombre de la ciudad: ')
temperatura = float(input('Ingresa la temperatura promedio de la ciudad (°C): '))
precipitacion = float(input("Ingresa la precipitación promedio de la ciudad (mm): "))

ciudad_a_predecir = pd.DataFrame([[temperatura, precipitacion]], columns=['temperatura_promedio', 'precipitacion_promedio'])

prediccion = modelo.predict(ciudad_a_predecir)
print(f'\nEl tipo de clima para la ciudad de {ciudad} es {"\033[38;5;208m"}{prediccion[0].capitalize()}{"\033[0m"}')
