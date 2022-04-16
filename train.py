#importar librerías

import os
import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt


from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

import warnings
warnings.filterwarnings('ignore')


#generar el dataframe a partir del archivo limpio

df = pd.read_csv(r"C:\Users\hugom\OneDrive\Documents\The Bridge_Data_Science\Alumno\3-Machine_Learning\Entregas\github\processed\store_cleaned.csv", index_col=0)


#definir las features y dividir en train y test

X = df[['Price', "Profit", "Quantity", "Segment", "Category", "Sub-Category"]]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size   = 0.2,
                                        random_state = 42)


#construir y entrenar el mejor modelo - XGBRegressor

xgb = XGBRegressor()

xgb.fit(X_train, y_train)

xgb.predict(X_test)

#evaluamos el modelo

#MAE MSE RMSE

print('MAE test', mean_absolute_error(y_test, xgb.predict(X_test)))
print('MSE test', mean_squared_error(y_test, xgb.predict(X_test)))
print('RMSE test', np.sqrt(mean_squared_error(y_test, xgb.predict(X_test))))

#Score

xgb.score(X_test, y_test)


#Generamos un grafico de predcción

xgb_predict = xgb.predict(X_test)

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, xgb_predict, label="predicted")
plt.title("Store test and predicted data")
plt.legend()
plt.show()


#guardamos el modelo final

import pickle

filename = r"C:\Users\hugom\OneDrive\Documents\The Bridge_Data_Science\Alumno\3-Machine_Learning\Entregas\github\model\final\xgb_model"

with open(filename, 'wb') as archivo_salida:
    pickle.dump(xgb, archivo_salida) 
