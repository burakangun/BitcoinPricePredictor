from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv('bitcoin_training.csv')
'''
#Test
print(datas)
'''
open = datas.iloc[:, 1].values
rest = datas.iloc[:, 3:6].values
high = datas.iloc[:, 2].values
volume = datas.iloc[:, 6].values

# print(volume)

# DataFrameleri Oluşturma
result = pd.DataFrame(data=open, index=range(2747), columns=['Open'])
result2 = pd.DataFrame(data=rest, index=range(2747), columns=[
                       'Low', 'Close', 'Adj Close'])
result3 = pd.DataFrame(data=high, index=range(2747), columns=['High'])
result4 = pd.DataFrame(data=volume, index=range(2747), columns=['Volume'])


# DataFrameleri Birleştirme
r = pd.concat([result, result2], axis=1)
ra = pd.concat([r, result4], axis=1)
r2 = pd.concat([ra, result3], axis=1)

print(r2)

# Verileri eğitim ve test için bölme

x_train, x_test, y_train, y_test = train_test_split(
    ra, result3, test_size=0.34, random_state=0)

# Verilerin ölçeklenmesi
# from sklearn.preprocessing import StandartScaler

# sc = StandartScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)

# Modeli eğitme
regressor = LinearRegression()
regressor.fit(x_train, y_train)
x_train = x_train.sort_index()
y_train = y_train.sort_index()

y_pred = regressor.predict(x_test)


#Grafiğe dökme
plt.plot(x_train, y_train) 
plt.xlim(0,90000)
plt.ylim(0,100000)
plt.show()
plt.plot(x_test,y_pred)
plt.xlim(0,90000)
plt.ylim(0,90000)
plt.show()


















