
#Batch normalization using keras


model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])

#Transfer learning for binary classsfier

model_A = keras.models.load_model("my_model_A.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])#slicing the last layer of the  model
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
# To clone the model
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())# weights should be added manually

# New output layer which is initialized randomly will make much error if trained again so we need to freez it
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
# after freezing and unfreeing model must be compiled
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
metrics=["accuracy"])


history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
validation_data=(X_valid_B, y_valid_B))
for layer in model_B_on_A.layers[:-1]:
layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))


