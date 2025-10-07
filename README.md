# Predicción con Redes Neuronales
Proyecto de algoritmo de Machine Learning

Este proyecto de ciencia de datos, analiza, calcula y predice la probabilidad de un cliente, si es de alto valor(En cuanto a pagos d edeudas y préstamos) o de bajo valor, y el desarrollo de una app web.

Primero importamos las librerías, las herramientas, luego, importamos el data set, con los datos para entrenar el modelo(se encuentra en Data, como "Clientes-RedesN.csv").

Realizamos luego una visualización del data set(convertido en dataframe), con diversos gráficos como gráfico de dispersión, matriz de correlación, etc(representando la relación entre las variables y copmo afecta al cliente).

Por siguiente creamos el modelo, lo entrenamos, y vemos cual es su rango de error de predicción, un gran detalle es que el modelo presenta sesgo por la clase mayoritaria(0=Bajo valor) y por ende tratamos este problema con la técnica de aumentar las épocas de entrenamiento y equilibrandolo con datos sintéticos al modelo. Le damos otro dato para predecir y por úlyimo identificamos el coeficiente e intercepto.

#Tencnologías: (Python: Numpy, Pandas, Matplotlib, Seaborn, Tensorflow y Scikit-learn) #Fuente de Datos: Dataset creado artificialmente en ChatGPT, con base a entidades financieras. (dataset: Clientes-RedesN.csv)
