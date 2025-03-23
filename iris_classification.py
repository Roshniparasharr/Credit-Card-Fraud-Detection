import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Load the Iris dataset from a CSV file into a pandas DataFrame
data = pd.read_csv(r'E:\work\Iris-flower-classification\iris.csv')

# Step 2: Explore the dataset
# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Display information about the dataset, including data types and non-null counts
print("\nDataset information:")
print(data.info())

# Display summary statistics for each feature in the dataset
print("\nSummary statistics:")
print(data.describe())

# Step 3: Check for missing values
# Check if there are any missing values in the dataset
print("\nMissing values in each column:\n", data.isnull().sum())

# Step 4: Encode the target variable
# Convert the categorical target variable 'species' into numerical labels
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Step 5: Visualize the data
# Create a pair plot to visualize relationships between features and the target variable
sns.pairplot(data, hue='species')
plt.show()

# Create a box plot to visualize the distribution of each feature
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, orient='h')
plt.show()

# Step 6: Analyze feature correlations
# Calculate the correlation matrix to understand relationships between features
corr_matrix = data.corr()
print("\nCorrelation matrix:\n", corr_matrix)

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Step 7: Split the data into training and testing sets
# Separate the features (X) from the target variable (y)
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the K-Nearest Neighbors (KNN) model
# Initialize the KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model using the training data
knn.fit(X_train, y_train)

# Step 9: Make predictions
# Use the trained model to make predictions on the test data
y_pred = knn.predict(X_test)

# Step 10: Evaluate the model
# Print the classification report to evaluate the model's performance
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix to see the model's accuracy in classifying each species
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))