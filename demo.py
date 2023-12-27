import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, w1_sd=1, w2_sd=1, activation_type='cos'):
        # Initialize weights and biases
        self.W1 = w1_sd * np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = w2_sd * np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.activation_type = activation_type
    
    def activation(self, x):
        # Activation function
        if self.activation_type == 'relu':
            return np.maximum(0, x)
        elif self.activation_type == 'cos':
            return np.cos(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")
    
    def dactivation(self, x):
        # Derivative of the activation function
        if self.activation_type == 'relu':
            return (x > 0).astype(float)
        elif self.activation_type == 'cos':
            return -np.sin(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_type}")
    
    def forward(self, x):
        # Forward pass through the network
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def get_loss(self, y_true, y_pred):
        # Calculate the loss
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, x, y_true, y_pred, learning_rate):
        m = x.shape[0]
        
        # Calculate gradients for output layer
        dz2 = (y_pred - y_true) / m
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Calculate gradients for hidden layer
        dz1 = np.dot(dz2, self.W2.T) * self.dactivation(self.z1)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    # Train the neural network
    def train(self, x, y, epochs, learning_rate, verbose=True):
        losses = []

        for epoch in range(epochs):
            n = x.shape[0]

            # For each pass over the training dataset
            for i in range(n):

                # Get a random sample from the training dataset
                random_idx = np.random.randint(n)
                xi = x[random_idx:random_idx+1]
                yi = y[random_idx:random_idx+1]

                # Perform forward and backward passes
                y_pred = self.forward(xi)
                self.backward(xi, yi, y_pred, learning_rate)
            
            # Calculate the loss for this epoch
            y_preds = self.forward(x)
            epoch_loss = self.get_loss(y, y_preds)
            losses.append(epoch_loss)

            # Print the loss for this epoch
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
            
        return losses

def train_network(x, y, w1_sd=10, w2_sd=0.1, epochs=100, learning_rate=0.01, activation_type='cos', plot=True):
    nn = NeuralNetwork(input_size=1, hidden_size=32, output_size=1, w1_sd=w1_sd, w2_sd=w2_sd, activation_type=activation_type)
    losses = nn.train(x, y, epochs=epochs, learning_rate=learning_rate, verbose=False)

    if plot:
        # Create subplots for loss progression and predictions
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the loss progression
        axs[0].plot(losses)
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title(f"Final loss: {losses[-1]:.5f}")

        # Plot the predictions against the actual data
        y_preds = nn.forward(x)

        x = x.flatten()
        y_preds = y_preds.flatten()
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        y_preds = y_preds[sorted_indices]

        axs[1].scatter(x, y, label='Actual', alpha=0.5, s=20)
        axs[1].plot(x, y_preds, label='Predicted', c='orange')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].set_title('Actual vs predicted')
        axs[1].legend()

        # Show the subplots
        plt.show()
        
    return losses

epochs = 100
learning_rate = 0.01
n_samples = 100

x = np.linspace(0, 1, n_samples).reshape(-1, 1)
y = np.exp(- 50 * (x - 0.5) ** 2)

# Other possible target functions
# y = 2 * x + 1
# y = x * np.sin(4 * np.pi * x)
# y = np.piecewise(x, [x < 0.5, x >= 0.5], [lambda x: x, lambda x: 0.5-np.sin(8*np.pi*x)])
# y = 2 * np.sqrt(1 - x ** 2)
# y = np.piecewise(x, [x < 0, x >= 0], [lambda x: 0, lambda x: 1])

train_network(
    x=x, 
    y=y, 
    w1_sd=10, 
    w2_sd=0.1, 
    epochs=epochs,
    learning_rate=learning_rate,
    activation_type='cos', 
    plot=True
)