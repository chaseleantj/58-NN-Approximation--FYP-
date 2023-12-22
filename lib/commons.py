import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import copy

class Layer_Input:
    def forward(self, inputs, training=False):
        self.output = inputs

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0, weight_sd=1, bias_sd=1):
        # self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # self.biases = np.random.randn(1, n_neurons)

        # self.weights = np.random.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_neurons))
        # self.biases = np.zeros((1, n_neurons))

        self.weights = weight_sd * np.random.randn(n_inputs, n_neurons)
        self.biases = bias_sd * np.random.randn(1, n_neurons)

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    
    def forward(self, inputs, training=False):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_Linear:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs
        
class Activation_ReLU:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs
    
class Activation_Cosine:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = np.cos(inputs)
    def backward(self, dvalues):
        self.dinputs = -np.sin(self.inputs) * dvalues
    def predictions(self, outputs):
        return outputs

class Activation_Sigmoid:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Softmax:
    def forward(self, inputs, training=False):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization=False):
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights ** 2)
                
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases ** 2)
        return regularization_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Accuracy_Binary(Accuracy):
    def init(self, y):
        pass
    def compare(self, predictions, y):
        predictions = (predictions > 0.5) * 1
        return predictions == y

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None, visualizer=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
        self.visualizer = visualizer

    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X, training=False):
        self.input_layer.forward(X, training=training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training=training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def train(self, X, y, *, epochs=1, batch_size=None, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        if self.visualizer is not None:
            if self.visualizer.animate:
                self.visualizer.start_graph()

        for epoch in tqdm(range(1, epochs+1)):

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step+1) * batch_size]
                    batch_y = y[step * batch_size:(step+1) * batch_size]
            
                output = self.forward(batch_X, training=True)
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
            train_loss = self.loss.calculate_accumulated(include_regularization=False)
            train_accuracy = self.accuracy.calculate_accumulated()

            if validation_data is not None:
                validation_loss, validation_accuracy = self.evaluate(*validation_data, batch_size=batch_size)

            if self.visualizer is not None:
                self.visualizer.train_loss_history.append(train_loss)
                self.visualizer.train_accuracy_history.append(train_accuracy)
                if self.visualizer.validation and validation_data is not None:
                    self.visualizer.test_loss_history.append(validation_loss)
                    self.visualizer.test_accuracy_history.append(validation_accuracy)
                if self.visualizer.animate and (epoch % self.visualizer.period == 0 or epoch == 1):
                    self.visualizer.update_graph(model=self, clear=True)
        if self.visualizer is not None:
            self.visualizer.update_graph(model=self, clear=False)
            plt.show()
    
    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                batch_y = y_val[step * batch_size : (step + 1) * batch_size]
            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated(include_regularization=False)
        validation_accuracy = self.accuracy.calculate_accumulated()
        return validation_loss, validation_accuracy
    
    def predict(self, X, *, batch_size=None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size : (step + 1) * batch_size]
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)
        return np.vstack(output)

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_parameters(self, path):
        with open(path, "rb") as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        for layer in model.layers:
            for property in ["inputs", "output", "dinputs", "dweights", "dbiases"]:
                layer.__dict__.pop(property, None)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
            return model

class Visualizer:
    def __init__(self, X, y, period=10, animate=False, validation=False, graph=None, graph_resolution=100):
        
        self.X = X
        self.y = y
        self.period = period
        self.animate = animate
        self.graph = graph
        self.graph_resolution = graph_resolution
        self.validation = validation
        self.train_accuracy_history = []
        self.train_loss_history = []

        if self.validation:
            self.test_accuracy_history = []
            self.test_loss_history  = []
        if self.graph in ["regression1d", "regression2d", "categorical2d"]:
            self.figure = plt.figure()
            self.figure.suptitle("Training neural network")
            gs = self.figure.add_gridspec(2, 3)
            self.axes = []
            self.axes.append(self.figure.add_subplot(gs[0,2]))
            self.axes.append(self.figure.add_subplot(gs[1,2], sharex=self.axes[0]))
            if self.graph == "regression2d":
                self.axes.append( self.figure.add_subplot(gs[:,0:2], projection="3d"))
            else:
                self.axes.append( self.figure.add_subplot(gs[:,0:2]))
            self.axes[0].tick_params(labelbottom=False, bottom=False)
            self.axes[0].yaxis.tick_right()
            self.axes[1].yaxis.tick_right()
        else:
            self.figure, self.axes = plt.subplots(2, sharex=True)
            self.figure.suptitle("Training neural network")
            self.axes[0].tick_params(labelbottom=False, bottom=False)

    
    def start_graph(self):
        plt.show(block=False)

    
    def update_graph(self, model, clear=False):
        history_len = len(self.train_accuracy_history)
        x_scale = range(1, history_len + 1)
        self.axes[0].set_ylim(0, 1)
        self.axes[0].plot(x_scale, self.train_accuracy_history, label="Training")
        self.axes[1].plot(x_scale, self.train_loss_history, label="Training")

        if self.validation:
            self.axes[0].plot(x_scale, self.test_accuracy_history, label="Validation")
            self.axes[1].plot(x_scale, self.test_loss_history, label="Validation")

        self.axes[0].set_title("Accuracy")
        self.axes[1].set_title("Loss")
        self.axes[0].legend(loc="lower right")
        self.axes[1].set_xlabel("Epochs")

        self.axes[0].grid(True)
        self.axes[1].grid(True)

        if self.graph in ["regression1d", "regression2d"]:
            self.axes[2].set_title("Regression")

            if self.graph == "regression1d":

                A = np.linspace(np.min(self.X), np.max(self.X), self.graph_resolution)
                B = model.forward(np.expand_dims(A, axis=-1), training=False).flatten()

                self.axes[2].scatter(self.X, self.y, s=1, c="black", alpha=0.5, label="Target")
                self.axes[2].plot(A, B, label="Fitted", c="red", linewidth=2)
                self.axes[2].legend()

            elif self.graph == "regression2d":

                a = np.linspace(np.min(self.X[:,0]), np.max(self.X[:,0]), self.graph_resolution)
                b = np.linspace(np.min(self.X[:,1]), np.max(self.X[:,1]), self.graph_resolution)
                A, B = np.meshgrid(a, b)
                predictions = model.forward(np.column_stack((A.flatten(), B.flatten())), training=False).reshape(1, -1)[0]
                C = np.reshape(predictions, A.shape)

                self.axes[2].scatter(self.X[:,0], self.X[:,1], self.y.flatten(), c="black", s=1)
                # self.axes[2].plot_trisurf(self.X[:,0], self.X[:,1], self.y.flatten(), cmap="magma", linewidth=0.5, edgecolors="black", alpha=0)
                # self.axes[2].plot_trisurf(self.X[:,0], self.X[:,1], np.zeros_like(self.y.flatten()), cmap="magma", linewidth=0.1, edgecolors="black", alpha=0.1)
                self.axes[2].plot_surface(A, B, C, cmap="magma", alpha=0.7, vmin=np.min(predictions), vmax=np.max(predictions) * 1.5)
                self.axes[2].grid(False)

        if self.graph == "categorical2d":
            size = np.min(self.X) * 1.2
            colors = np.array(sns.color_palette("husl", len(set(self.y)))) * 255
            palette = colors
            cmap = create_colormap(np.copy(colors), bit=True)
            filter = self.draw_pixels(x_range=(-size, size), y_range=(-size, size))
            filter = np.array([model.forward(row, training=False) for row in filter])
            product = np.matmul(filter, palette)
            img = np.array(product, dtype=int).reshape(self.graph_resolution, self.graph_resolution, 3)
            self.axes[2].imshow(img, extent=[-size, size, -size, size], alpha=0.9)
            self.axes[2].scatter(self.X[:,0], self.X[:,1], c=self.y, cmap=cmap, edgecolors="k", linewidths=0.5)
            self.axes[2].set_title("Visualization")

        self.figure.canvas.draw()
        if clear:
            self.figure.canvas.flush_events()
            for ax in self.axes:
                ax.cla()

    def draw_pixels(self, x_range, y_range):
        x_width = x_range[1] - x_range[0]
        y_width = y_range[1] - y_range[0]
        pixels = np.array([[[x_range[0] + i * x_width / self.graph_resolution, 
                            y_range[1] - j * y_width / self.graph_resolution] 
                            for i in range(self.graph_resolution)] for j in range(self.graph_resolution)])
        return pixels

def shuffle_array(arr1, arr2):
    arr1_copy = arr1.copy()
    arr2_copy = arr2.copy()
    keys = np.array(range(len(arr1)))
    np.random.shuffle(keys)
    shuffled_arr1 = arr1_copy[keys]
    shuffled_arr2 = arr2_copy[keys]
    return shuffled_arr1, shuffled_arr2

# Fixed a bug in Keras: https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
# Standardizes grayscale image vectors of shape (n x W * H)
def standardize_images(arr):
    # flattened = np.reshape(arr, (arr.shape[0], arr.shape[1], -1)) # for RGB images
    flattened = np.reshape(arr, (arr.shape[0], -1))
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std[std == 0] = 1
    return np.reshape(((flattened - mean) / std), np.shape(arr))

# Create colormap
# Source: https://github.com/CSlocumWX/custom_colormap/blob/master/custom_colormaps/custom_colormaps.py
def create_colormap(colors, position=None, bit=False, reverse=False, name='custom_colormap'):
    """
    returns a linear custom colormap
    Parameters
    ----------
    colors : array-like
        contain RGB values. The RGB values may either be in 8-bit [0 to 255]
        or arithmetic [0 to 1] (default).
        Arrange your tuples so that the first color is the lowest value for the
        colorbar and the last is the highest.
    position : array like
        contains values from 0 to 1 to dictate the location of each color.
    bit : Boolean
        8-bit [0 to 255] (in which bit must be set to
        True when called) or arithmetic [0 to 1] (default)
    reverse : Boolean
        If you want to flip the scheme
    name : string
        name of the scheme if you plan to save it
    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        cmap with equally spaced colors
    """
    from matplotlib.colors import LinearSegmentedColormap
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors, dtype='f')
    if reverse:
        colors = colors[::-1]
    if position is not None and not isinstance(position, np.ndarray):
        position = np.array(position)
    elif position is None:
        position = np.linspace(0, 1, colors.shape[0])
    else:
        if position.size != colors.shape[0]:
            raise ValueError("position length must be the same as colors")
        elif not np.isclose(position[0], 0) and not np.isclose(position[-1], 1):
            raise ValueError("position must start with 0 and end with 1")
    if bit:
        colors[:] = [tuple(map(lambda x: x / 255., color)) for color in colors]
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return LinearSegmentedColormap(name, cdict, 256)