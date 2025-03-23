import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'E:\work\Credit-Card-Fraud-Detection\iris.csv')

# Explore the dataset
print("First few rows of the dataset:")
print(data.head())
print("\nDataset information:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in each column:\n", data.isnull().sum())

# Encode the target variable
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Visualize the data
sns.pairplot(data, hue='species')
plt.show()

# Boxplot to see the distribution of each feature
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, orient='h')
plt.show()

# Correlation matrix
corr_matrix = data.corr()
print("\nCorrelation matrix:\n", corr_matrix)

# Visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Split the data
X = data.drop('species', axis=1)
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))