import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.layers import Dense
from keras.layers import Flatten


#loads the data 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#flattens the 1d vector length to 784 

#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)

model = Sequential()

from ann_visualizer.visualize import ann_viz
ann_viz(model)


model.add(Flatten()) 

model.add(Dense(10, activation='relu',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))



model.compile(loss = "mse", 
optimizer = Nadam(lr = 0.0001),
metrics=['categorical_accuracy'])


history = model.fit(x_train, y_train,
                batch_size= 128,
                epochs= 50,
                verbose=1,
                validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

history_dict = history.history

print(history_dict.keys()) 

plt.plot(range(50), history_dict['loss'], label='Loss') 
plt.plot(range(50), history_dict['categorical_accuracy'], label='Accuracy') 
plt.plot(range(50), history_dict['val_categorical_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.legend()
plt.show()
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.show()