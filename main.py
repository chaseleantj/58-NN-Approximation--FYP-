import idx2numpy
from lib.commons import *
from lib.optimizers import *
from lib.datasets import *
from lib.convolutional import *

##############################
# Single variable regression
##############################

n_samples = 50
X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
y = np.exp(- X ** 2)

model = Model()
visualizer = Visualizer(X=X, y=y, period=10, animate=True, graph="regression1d", graph_resolution=1000)
model.add(Layer_Dense(1, 32, weight_sd=10))
model.add(Activation_Cosine())
model.add(Layer_Dense(32, 1, weight_sd=0.1))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_SGD(learning_rate=0.01), accuracy=Accuracy_Regression(), visualizer=visualizer)

model.finalize()
model.train(X, y, batch_size=1, epochs=100)

##############################
# MNIST Classification
##############################

# sample_ratio = 1
# train_loss_arr = []
# test_loss_arr = []
# train_acc_arr = []
# test_acc_arr = []

# # MNIST dataset: http://yann.lecun.com/exdb/mnist/
# # These files are not included in the repository
# # Download them from the link above and place them inside a directory "data" in the root of the repository
# X = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")[:int(60000 * sample_ratio)]
# X = (X.astype(np.float32) - 127.5 ) / 127.5
# y = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")[:int(60000 * sample_ratio)]
# X, y = shuffle_array(X, y)

# X_test = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")[:int(10000 * sample_ratio)]
# X_test = (X_test.astype(np.float32) - 127.5 ) / 127.5
# y_test = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")[:int(10000 * sample_ratio)]
# X_test = np.expand_dims(X_test, axis=1)

# model = Model()
# model.add(Layer_Flatten())
# model.add(Layer_Dense(784, 1024, weight_sd=0.1))
# model.add(Activation_Cosine())
# model.add(Layer_Dense(1024, 10, weight_sd=0.01))
# model.add(Activation_Softmax())

# visualizer = Visualizer(X=X, y=y, animate=True, period=1, validation=True, graph=None)
# model.set(visualizer=visualizer, loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_SGD(learning_rate=0.01), accuracy=Accuracy_Categorical())
# model.finalize()
# model.train(X, y, validation_data=(X_test, y_test), batch_size=32, epochs=20)

# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test loss:", loss)
# print("Test accuracy:", accuracy)