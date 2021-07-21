from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold

# load data
df_nse = pd.read_csv('NSE-TATAResult.csv')
df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)

# new_data = pd.DataFrame(index=range(0, len(df_nse)),
#                         columns=['Close'])
# for i in range(0, len(data)):
#     new_data['Close'][i] = data['Close'][i]

new_data = pd.DataFrame(index=range(0, len(df_nse)),
                        columns=['CloseRateOfChange'])
for i in range(0, len(data)):
    new_data['CloseRateOfChange'][i] = data['CloseRateOfChange'][i]


def chuyenDoi(array):
    result = []
    for element in array:
        result.append(element[0])
    result = np.array(result)
    return result


x = []
y = []
for i in range(0, len(new_data) - 987):
    xx = []
    for j in range(0, 987):
        xx.append(new_data.values[i + j])

    xx = chuyenDoi(xx)
    x.append(xx)
    y.append(new_data.values[i + 987])

y = chuyenDoi(y)
x = np.array(x)

X_test = []
for i in range(1, len(new_data) - 986):
    xxx = []
    for j in range(0, 987):
        xxx.append(new_data.values[i + j])

    xxx = chuyenDoi(xxx)
    X_test.append(xxx)
X_test = np.array(X_test)


# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.24)


xgbr = xgb.XGBRegressor(verbosity=0)

xgbr.fit(x, y)
# xgbr.fit(xtrain, ytrain)

# score = xgbr.score(xtrain, ytrain)
score = xgbr.score(x, y)
print("Training score: ", score)

# - cross validataion
# scores = cross_val_score(xgbr, xtrain, ytrain, cv=5)

# kfold = KFold(n_splits=10, shuffle=True)
# kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold)

ypred = xgbr.predict(X_test)

pickle.dump(xgbr, open("pima.pickle_rate_of_change.dat", "wb"))
loaded_model = pickle.load(open("pima.pickle_rate_of_change.dat", "rb"))
yLoad = loaded_model.predict(X_test)
predictions = [round(value) for value in yLoad]
print('yload: ', predictions)

x_ax = range(len(y))
x_ay = range(len(X_test))
plt.scatter(x_ax, y, s=5, color="blue", label="original")
plt.plot(x_ay, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()
