# ML & AI Prioritized Variables and Concepts

# Essential for ML/AI Workflows

# 1. Data Variables
# List Variable: Store and manipulate collections of data
data_list = [1, 2, 3, 4, 5]
print("List Variable:", data_list)

# Dictionary Variable: Manage key-value pairs, useful for datasets and configurations
data_dict = {"name": "John", "age": 42}
print("Dictionary Variable:", data_dict)

# Pandas Data Science Variable: Analyze and manipulate tabular data
import pandas as pd
# Example: Load and describe data from a CSV file
data = pd.read_csv("data.csv")
print("Pandas DataFrame Description:\n", data.describe())

# Numpy Array Variable: Handle numerical data efficiently
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
print("Numpy Array:", numpy_array)

# Scikit-learn Machine Learning Variable: Build and evaluate ML models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model Accuracy:", model.score(X_test, y_test))

# TensorFlow AI Variable: Develop and train deep learning models
import tensorflow as tf
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
nn_model.compile(optimizer="adam", loss="binary_crossentropy")
# Placeholder for training
# nn_model.fit(X_train, y_train, epochs=10)

# 2. Data Handling & Serialization
# JSON Variable: Serialize and deserialize data
import json
data = {"name": "John", "age": 42}
json_string = json.dumps(data)
print("Serialized JSON:", json_string)
print("Deserialized JSON:", json.loads(json_string))

# Pickle Serialization Variable: Save and load ML models
import pickle
model_filename = "model.pkl"
# Save model
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
# Load model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Loaded Model Accuracy:", loaded_model.score(X_test, y_test))

# 3. Data Visualization
# Matplotlib Plot Variable: Visualize data
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
plt.title("Data Visualization Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Seaborn Visualization Variable: Create advanced data visualizations
import seaborn as sns
sns.set(style="darkgrid")
sns.histplot(data['age'], kde=True)
plt.show()

# 4. Numerical & Statistical Analysis
# Numeric Variable: Perform calculations
number = 42
print("Numeric Variable:", number)

# Complex Variable: Store a complex number with real and imaginary parts
complex_number = 1 + 2j
print("Complex Number:", complex_number)

# Decimal Variable: Represent a fixed-point number with high precision
from decimal import Decimal
decimal_number = Decimal("3.14")
print("Decimal Number:", decimal_number)

# Math & Random Variables: Perform math operations and generate random numbers
import math
import random
print("Square Root of 16:", math.sqrt(16))
print("Random Integer (1-10):", random.randint(1, 10))

# 5. Data Preprocessing & Manipulation
# Boolean Variable: Handle logical conditions
is_valid = True
print("Boolean Variable:", is_valid)

# None Variable: Represent a lack of value
not_set = None
print("None Variable:", not_set)

# Advanced Topics: Asynchronous & Parallel Processing
# Thread Variable: Run tasks in parallel
import threading
def task():
    print("Hello from a separate thread!")
thread = threading.Thread(target=task)
thread.start()

# Process Variable: Execute processes concurrently
import multiprocessing
process = multiprocessing.Process(target=task)
process.start()

# Queue Variable: Share data between processes
queue = multiprocessing.Queue()
queue.put("Hello from a multiprocessing queue!")
print(queue.get())