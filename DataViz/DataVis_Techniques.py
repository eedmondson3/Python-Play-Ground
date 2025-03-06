# Critical Data Visualization Techniques for Data Analysts

# 1. Importing Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data for Visualization
# Creating a simple DataFrame for demonstration purposes
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Values': [10, 23, 45, 32, 17],
    'Groups': ['G1', 'G2', 'G1', 'G2', 'G1'],
    'Time': pd.date_range(start='2023-01-01', periods=5, freq='D')
})
print("Sample Data:\n", data)

# 2. Bar Plot
# Useful for comparing categorical data
data.plot(kind='bar', x='Category', y='Values', title='Bar Plot Example', color='skyblue')
plt.xlabel('Category')
plt.ylabel('Values')
plt.show()

# 3. Line Plot
# Ideal for showing trends over time or sequential data
plt.figure(figsize=(8, 6))
plt.plot(data['Time'], data['Values'], marker='o', linestyle='-', color='teal')
plt.title('Line Plot Example - Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# 4. Scatter Plot
# Great for showing relationships between two numerical variables
plt.figure(figsize=(8, 6))
plt.scatter(data['Values'], np.random.randint(1, 100, size=len(data)), color='coral')
plt.title('Scatter Plot Example')
plt.xlabel('Values')
plt.ylabel('Random Data')
plt.show()

# 5. Histogram
# Displays the distribution of a single numerical variable
plt.figure(figsize=(8, 6))
plt.hist(data['Values'], bins=5, color='purple', edgecolor='black')
plt.title('Histogram Example - Value Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# 6. Box Plot
# Shows the distribution of a dataset based on five summary statistics
plt.figure(figsize=(8, 6))
sns.boxplot(x='Groups', y='Values', data=data, palette='Set2')
plt.title('Box Plot Example - Grouped Data')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.show()

# 7. Heatmap
# Displays the correlation matrix of the dataset using color gradients
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Heatmap Example - Correlation Matrix')
plt.show()

# 8. Pie Chart
# Useful for showing proportions within a categorical dataset
plt.figure(figsize=(6, 6))
plt.pie(data['Values'], labels=data['Category'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pie Chart Example')
plt.show()

# 9. Pair Plot
# Creates a matrix of scatter plots for all numerical columns
sns.pairplot(data, diag_kind='kde', corner=True)
plt.suptitle('Pair Plot Example', y=1.02)
plt.show()

# 10. Violin Plot
# Combines aspects of box plot and density plot
plt.figure(figsize=(8, 6))
sns.violinplot(x='Groups', y='Values', data=data, palette='muted')
plt.title('Violin Plot Example')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.show()
