# Load librarY

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, SGDClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.externals import joblib

# Load ROSE processing data

doc = pd.read_csv('rose_data.csv', encoding='cp949')

# New dataframe copy exclude unnecessary column

data = doc[['풍속', '기압', '습도', '기온', '수온', '최대파고.m.', '유의파고.m.', '평균파고.m.', '파주기.sec.', 'check']].copy()

# Variable set

x = np.array(data[['풍속', '기압', '습도', '기온', '수온', '최대파고.m.', '유의파고.m.', '평균파고.m.', '파주기.sec.']])
y = np.array(data[['check']])

# Scaling: MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
min_max_scaling_x = scaler.transform(x)

# Test, Train data set test_szie: 0.30
# test_size : 테스트 데이터셋의 비율(float)이나 갯수(int) (default = 0.25)

xtrain, xtest, ytrain, ytest = train_test_split(min_max_scaling_x, y, test_size=0.30, random_state=42)

# Set alpha value

alphas = np.logspace(-4, 0, 200)
parameters = {'alpha': alphas }

# Find optimal alpha value
# Solve warning msg -> .ravel()
# '풍속', '기압', '습도', '기온', '수온', '최대파고.m.', '유의파고.m.', '평균파고.m.', '파주기.sec.'

elastic_cv=ElasticNetCV(alphas=alphas, cv=5)
model = elastic_cv.fit(xtrain, ytrain.ravel())
print(model.alpha_)
print(model.coef_)
print(model.intercept_)

# Check Oods ratio

print(np.exp(model.coef_))

# Use SGDClassifier
# Solve warning msg -> .ravel()

elastic=SGDClassifier(loss="hinge", max_iter=100000, penalty="elasticnet", tol=0.01, random_state=42).fit(xtrain, ytrain.ravel())

# Check score, MSE, RMSE

ypred = elastic.predict(xtest)
score = elastic.score(xtest, ytest)
mse = mean_squared_error(ytest, ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}".format(score, mse, np.sqrt(mse)))

# Check result

cr = classification_report(ytest, ypred)
print(cr)

# Check confusion_matrix

confusion_matrix(ytest, ypred)

# Load model

joblib.dump(elastic, '../model/model.pkl')