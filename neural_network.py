import numpy as np

class ThreeLayerNN:
    def __init__(self, input_dim=10, hidden_units=8, output_dim=1, learning_rate=0.01):
        # Initialize network parameters
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        # Input layer -> Hidden layer 1
        self.W1 = np.random.randn(input_dim, hidden_units) * np.sqrt(2.0/input_dim)  # He initialization
        self.b1 = np.zeros((1, hidden_units))
        
        # Hidden layer 1 -> Hidden layer 2
        self.W2 = np.random.randn(hidden_units, hidden_units//2) * np.sqrt(2.0/hidden_units)
        self.b2 = np.zeros((1, hidden_units//2))
        
        # Hidden layer 2 -> Output layer
        self.W3 = np.random.randn(hidden_units//2, output_dim) * np.sqrt(2.0/(hidden_units//2))
        self.b3 = np.zeros((1, output_dim))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        # Forward propagation
        self.A0 = X  # Input layer
        
        # Hidden layer 1
        self.Z1 = np.dot(self.A0, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Hidden layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        
        # Output layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)
        
        return self.A3
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Output layer
        dZ3 = output - y
        dW3 = (1/m) * np.dot(self.A2.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        
        # Hidden layer 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        epsilon = 1e-15  # Small constant to avoid log(0)
        return -1/m * np.sum(y_true * np.log(y_pred + epsilon) + 
                           (1 - y_true) * np.log(1 - y_pred + epsilon))
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        m = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward propagation
                output = self.forward(X_batch)
                
                # Backward propagation
                self.backward(X_batch, y_batch, output)
            
            # Compute loss for the entire dataset
            predictions = self.forward(X)
            loss = self.compute_loss(y, predictions)
            losses.append(loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses 