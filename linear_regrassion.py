import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from google.colab import drive
drive.mount('/content/drive/',force_remount=True)

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/drive/MyDrive/ML_LAB/Salary_Data.csv")

data.head(20)

plt.scatter(data['YearsExperience'], data['Salary'])
plt.show()

corr = data.corr()
corr

train_data, test_data = train_test_split(data, test_size=.2, random_state=42)
train_x, train_y = np.array(train_data['YearsExperience']).reshape(-1,1),np.array(train_data['Salary']).reshape(-1,1)
test_x, test_y = np.array(test_data['YearsExperience']).reshape(-1,1),np.array(test_data['Salary']).reshape(-1,1)
print(train_y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
train_x

model = LinearRegression()model.score(test_x,test_y)


model.fit(train_x, train_y)


model.score(train_x,train_y)

score = cross_val_score(model, train_x,train_y,scoring='neg_mean_absolute_error',cv=3)
score = (-score)
score.mean()

test_x = scaler.transform(test_x)
test_predict = model.predict(test_x)
score = mean_absolute_error(test_y,test_predict)
score

model.score(test_x,test_y)

plt.scatter(test_x,test_y,color = 'b',alpha = 0.8)
plt.plot(test_x,test_predict,color='r')
plt.show()
