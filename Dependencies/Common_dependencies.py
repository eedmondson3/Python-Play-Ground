# Most Commonly Used Dependencies and Imports for Data Analysts & Data Scientists

# 1. Data Manipulation and Analysis
import pandas as pd  # DataFrame manipulation and analysis
import numpy as np  # Numerical operations and array processing

# 2. Data Visualization
import matplotlib.pyplot as plt  # Basic plotting and visualization
import seaborn as sns  # Advanced visualization and statistical plots

# 3. Machine Learning and AI
from sklearn.model_selection import train_test_split  # Splitting datasets
from sklearn.ensemble import RandomForestClassifier  # Classification models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # Evaluation metrics
import tensorflow as tf  # Deep learning and neural networks

# 4. Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Data scaling and encoding
from sklearn.impute import SimpleImputer  # Handling missing data

# 5. Statistical Analysis
from scipy import stats  # Statistical tests and distributions

# 6. Handling Dates and Times
from datetime import datetime, timedelta  # Date and time manipulation
import time  # Handling time-based operations

# 7. Working with Files and Directories
import os  # Interacting with the operating system
from pathlib import Path  # Handling file paths

# 8. Web Scraping and API Requests
import requests  # Making HTTP requests
from bs4 import BeautifulSoup  # Parsing HTML and web scraping
import json  # Handling JSON data

# 9. Data Storage and Databases
import sqlite3  # Working with SQLite databases
import pickle  # Serializing and deserializing Python objects

# 10. Regular Expressions for Data Cleaning
import re  # Matching patterns in strings

# 11. Automation and Scripting
import subprocess  # Running shell commands
import logging  # Setting up logging for scripts

# 12. Advanced Data Visualization (Interactive)
import plotly.express as px  # Simple and powerful plotting
import plotly.graph_objects as go  # Advanced graphing with Plotly
from bokeh.plotting import figure, output_file, show  # Interactive visualizations

# 13. Asynchronous Programming
import asyncio  # Handling asynchronous tasks

# 14. Parallel and Multithreading
import threading  # Managing multiple threads
import multiprocessing  # Running multiple processes

# 15. Configurations and Environment Variables
import configparser  # Handling configuration files
print(os.getenv("USER"))  # Accessing environment variables

# 16. Mathematical Operations
import math  # Performing mathematical calculations
import random  # Generating random numbers

# 17. Testing and Validation
import unittest  # Unit testing framework

# 18. Network Communication
import socket  # Creating and managing network connections

# 19. Handling Compressed Files
import zipfile  # Compressing and decompressing zip files
import zlib  # Working with compressed data

# 20. Cryptography and Security
from cryptography.fernet import Fernet  # Encryption and decryption

# Example Usage of Common Imports
# Pandas and Numpy
data = pd.DataFrame(np.random.rand(10, 5), columns=[f'col{i}' for i in range(5)])
print("Data Sample:\n", data.head())

# Visualization Example
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Machine Learning Example
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['col0']), data['col0'], test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
