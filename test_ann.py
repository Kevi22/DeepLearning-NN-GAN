import numpy as np 
from neural_network import ThreeLayerNN

# Create sample data
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, (100, 1))  # Binary classification

# Create and train the model
nn = ThreeLayerNN(input_dim=10, hidden_units=8, output_dim=1, learning_rate=0.01)

# Train the model
losses = nn.train(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = nn.forward(X_train)
predictions_binary = (predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions_binary == y_train)
print(f"\nFinal accuracy: {accuracy:.4f}") 