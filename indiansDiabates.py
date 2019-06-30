# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:25:24 2019

@author: saket
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

#from keras.utils import plot_model


dataset = numpy.loadtxt("pima-indians-diabetes.csv",delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y,epochs=150,batch_size=10)


#making predictions
predictions = model.predict(X)
results = [round(result[0]) for result in predictions] # it gives arrays of array therefore result[0]


# evaluate the model
loss, score = model.evaluate(X, y)

print("\n%s: %.2f%%" % (model.metrics_names[1], score*100))

