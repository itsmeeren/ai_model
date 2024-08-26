# imports
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.regularizers import l1


# creating the object of dataset

cifar_images = keras.datasets.cifar10
#loading the dataset into training and testing data

(x_train,y_train),(x_test,y_test) = cifar_images.load_data()

#data splitting

import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('dark_background')
# for x in x_train[:5]:
#   plt.imshow(x)
#   plt.show()

# normalizing the image values range from 0-1
x_train=x_train/255
x_test=x_test/255

print(x_train.shape)
print(x_test.shape)


#without conv2D and maxpooling

# Define the model
model = Sequential()

# Flatten layer for converting input to 1D
model.add(Flatten())

# Add Dense layers with He initializer, L1 regularization, and Dropout layers
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.01)))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.01)))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.01)))
model.add(Dropout(0.5))

# Output layer with softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)



# with other modification

model = Sequential()
model.add(Flatten())

# Adding Batch Normalization and tweaking Dropout rate
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))

# Compile with Nadam optimizer
model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


# with maxpooling and conv2d
input_shape = x_train.shape[1:]

model = Sequential()

# Conv2D layer with input_shape derived from x_train
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=HeNormal(), input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Further layers...
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

import scikeras
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Nadam

# Define the model function
def create_model(learning_rate=0.001, kernel_initializer='he_normal', l1_reg=0.001):
    model = Sequential()
    input_shape = x_train.shape[1:]  # Set the input shape

    # Select the appropriate kernel initializer
    if kernel_initializer == 'he_normal':
        init = HeNormal()
    else:
        init = GlorotUniform()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=init, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer=init, kernel_regularizer=l1(l1_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu', kernel_initializer=init, kernel_regularizer=l1(l1_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Nadam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'batch_size': [16, 32],
    'epochs': [10, 20],
    'model__learning_rate': [0.001, 0.01],  # Corrected prefix
    'model__l1_reg': [0.001, 0.01],          # Adjusted l1 regularization parameter
    'model__kernel_initializer': ['he_normal', 'glorot_uniform']  # Corrected prefix
}

# Perform Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



# with random search
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


# Define the model function
def create_model(learning_rate=0.001, kernel_initializer='he_normal', l1_reg=0.001):
    model = Sequential()
    input_shape = x_train.shape[1:]  # Set the input shape

    # Select the appropriate kernel initializer
    if kernel_initializer == 'he_normal':
        init = HeNormal()
    else:
        init = GlorotUniform()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=init, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer=init, kernel_regularizer=l1(l1_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu', kernel_initializer=init, kernel_regularizer=l1(l1_reg)))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Nadam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the hyperparameter grid
param_dist = {
    'batch_size': [16, 32],
    'epochs': [10, 20],
    'model__learning_rate': [0.001, 0.01],
    'model__l1_reg': [0.001, 0.01],
    'model__kernel_initializer': ['he_normal', 'glorot_uniform']
}

# Perform Randomized Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=42)
random_result = random_search.fit(x_train, y_train)

# Print the best parameters and score
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))



'''
Glarot initializer
droout reduced to 0.2
and performance scheduling ( ReduceLROnPlateau )
optimizer adam




'''

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau

input_shape = x_train.shape[1:]

model = Sequential()

# Conv2D layer with input_shape derived from x_train
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=GlorotNormal(), input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Further layers...
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=GlorotNormal()))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer=GlorotNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_initializer=GlorotNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Fit the model with the learning rate scheduler
model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[lr_scheduler])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Save the model
model.save('my_cifar10_model.h5')  # Save the model in HDF5 format



from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam

input_shape = x_train.shape[1:]

model = Sequential()

# First Conv Layer
model.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer=GlorotNormal(), input_shape=input_shape, padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Conv Layer
model.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer=GlorotNormal(), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Conv Layer
model.add(Conv2D(128, (3, 3), activation='elu', kernel_initializer=GlorotNormal(), padding='same'))
model.add(BatchNormalization())

# Fourth Conv Layer (New)
model.add(Conv2D(256, (3, 3), activation='elu', kernel_initializer=GlorotNormal(), padding='same'))
model.add(BatchNormalization())

# Fifth Conv Layer (New)
model.add(Conv2D(256, (3, 3), activation='elu', kernel_initializer=GlorotNormal(), padding='same'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))  # Add pooling layer after the last conv layer

model.add(Flatten())

model.add(Dense(128, activation='elu', kernel_initializer=GlorotNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.2))

model.add(Dense(64, activation='elu', kernel_initializer=GlorotNormal(), kernel_regularizer=l1(0.001)))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

# Compile the model with Nadam optimizer
model.compile(optimizer=Nadam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# Fit the model with the learning rate scheduler
model.fit(x_train, y_train, epochs=30, validation_split=0.2, callbacks=[lr_scheduler])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save('my_cifar10_model_v2.h5')  # Save the modified model


#testing with unseen data
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img_path = '/content/car.png'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(32, 32))  # Resize to match input shape
img_array = image.img_to_array(img)  # Convert to array
img_array = img_array / 255.0  # Normalize if necessary
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)
