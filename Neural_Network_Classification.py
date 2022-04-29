import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/deep learning/dataset/diabetes (2).csv")
data.head(10)


data.tail(10)

data.columns
data.values

dataset=data.values
X=dataset[:,0:8]
Y=dataset[:,8]
print(X)
print(Y)

from sklearn import preprocessing
min_max_scaler=preprocessing.MinMaxScaler()
X_scale=min_max_scaler.fit_transform(X)
X_scale

from keras.utils import np_utils

encoded_Y=np_utils.to_categorical(Y)
encoded_Y



from sklearn.model_selection import train_test_split
X_training,X_testing,Y_training,Y_testing = train_test_split(X_scale,encoded_Y,test_size=0.2,random_state=10)
X_training,X_valid,Y_traning,Y_valid = train_test_split(X_training,Y_training,test_size=0.2,random_state=10)
print(len(X_training))
print(len(Y_training))
print(len(X_testing))
print(len(X_valid))



from keras.models import Sequential
from keras.layers import Dense

#creating the model
model = Sequential()
model.add(Dense(24,input_shape=(8,), activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(12,activation='tanh'))
model.add(Dense(8,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()#give a summary of madel

!pip3 install tensorflow
!pip3 install keras


from tensorflow.keras import optimizers
opt=optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])




hist = model.fit(X_training,Y_training,batch_size=4,epochs=5,validation_data=(X_valid,Y_valid))

import matplotlib.pyplot as plt 
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('custom_trainvalacc.png')
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
#plt.show()
plt.savefig('custom_trainvalloss.png')
plt.figure()
