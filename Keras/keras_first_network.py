# Undergraduate Student: Arturo Burgos
# Professor: Aldemir Cavalini Jr and Aristeu da Silveira Neto
# Federal University of Uberlândia - UFU, Fluid Mechanics Laboratory - MFLab, Block 5P, Uberlândia, MG, Brazil

# first neural network with keras tutorial 

from numpy import loadtxt 
from keras.models import Sequential
from keras.layers import Dense


# load the dataset ('pima-indians-diabetes.csv')

dataset = loadtxt('/home/arturo/Deep-Learning/Keras/pima-indians-diabetes.csv', delimiter=',')
# delimiter is the char which delimites the fuction to read the data


# split into input (X) and output (y) variables 
# Data will be stored in a 2D array where the first dimension is the .txt row
# and the second dimension is columns

###################################################################################################################################
# manipulate X and y 
#X = dataset[0,0:8]
#y = dataset[0,7]

#print(X)
#print('\n')
#print(y)
###################################################################################################################################

X = dataset[:,0:8] # We do not reach to the 9th element to do so 0:9
y = dataset[:,8] # Here we reach to the 9th element, because python begins with 0,1,2,...,n

# Models in Keras are defined as a sequence of layers, just like the LSTM or the HTM
# We create a Sequential model and add layers one at a time until we are happy with our network architecture.

# We must to certified that the input layer has the right number of input features
# How do we know the number of layers and their types?
# The best network structure is found through a process of trial and error experimentation
# Generally, you need a network large enough to capture the structure of the problem.
# In this example, we will use a fully-connected network structure with three layers.

# instance  --> Defining the keras model layers
# - The model expects rows of data with 8 variables (the input_dim=8 argument) --> X?
# - The first hidden layer has 12 nodes and uses the relu activation function <-- is this the input layer and why 12?
# - The second hidden layer has 8 nodes and uses the relu activation function
# - The output layer has one node and uses the sigmoid activation function

model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

###################################################################################################################################

#Note, the most confusing thing here is that the shape of the input to the model is defined as an argument
#on the first hidden layer. This means that the line of code that adds the first Dense layer is doing 2
#things, defining the input or visible layer and the first hidden layer.

###################################################################################################################################

# Now we compile it by choosing a backend device, in that case is the TensorFlow
# The beckend choose the best way to represent the network traning of making predictions

# compile the keras model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# DISCUSS WITH DENISE_@_@_@_@_@_@_@_@_@_@_@_@_@_@__@_@_@_@_@_@__@_@_@__@_@_@_@__@_@__@_@_@_@__@_@_@_@_@_@_@__@_@_@_@_@_@_@__@_@_@_@

# epochs --> training time and batch_size --> samples from the epoch before weitghts are update
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)

#predictions = model.predict(X)
# round predictions 
#rounded = [round(x[0]) for x in predictions]


# make probability predictions with the model
predictions = model.predict_classes(X)
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))














