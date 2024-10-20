import ray
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Initialize Ray cluster
ray.init(address='auto')  # Connect to the Ray cluster

# Function to simulate data generation and distribution
def generate_data(num_samples, num_features, num_servers):
    """Simulate generating a distributed dataset across multiple servers."""
    datasets = []
    samples_per_server = num_samples // num_servers
    
    for _ in range(num_servers):
        X, y = make_classification(n_samples=samples_per_server, n_features=num_features, random_state=42)
        datasets.append((X, y))
    
    return datasets

# Remote function to train a model
@ray.remote
def train_model(X_train, y_train):
    model = SGDClassifier(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    return model

# Remote function to evaluate a model
@ray.remote
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    performance = accuracy_score(y_test, y_pred)
    return performance

# Main function to manage distributed training
def distributed_training(num_servers=4, num_samples=10000, num_features=20):
    # Generate distributed dataset
    distributed_data = generate_data(num_samples, num_features, num_servers)
    
    # Prepare the test dataset (can be a separate dataset or one of the training partitions)
    X_test, y_test = make_classification(n_samples=2000, n_features=num_features, random_state=42)
    
    # Train models in parallel on each server's dataset
    futures = [train_model.remote(X, y) for X, y in distributed_data]
    
    # Collect trained models
    models = ray.get(futures)
    
    # Evaluate each model and collect performance
    eval_futures = [evaluate_model.remote(models[i], X_test, y_test) for i in range(num_servers)]
    performances = ray.get(eval_futures)
    
    # Output performance of each model
    for i, performance in enumerate(performances):
        print(f"Model {i + 1} Performance: {performance:.4f}")

# Run the distributed training
distributed_training()
