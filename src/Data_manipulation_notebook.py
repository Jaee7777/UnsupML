import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

"""
for this dataset, I already know some information about it , i.e. by visiting UCI repository, however
if I did not have any information, I could read the dataset first to investigate it and them make more
informed ideas about it. Running the following will show the first few rows of the dataset

What do you see?

df_preview = pd.read_csv(url, header=None)
print("First few rows of the dataset:")
print(df_preview.head())

"""

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
df = pd.read_csv(url, names=columns, na_values=" ?")

# Display first few rows
print(df.head())  #  Inspect the dataset

# Show dataset dimensions
print(df.info())

# Show missing values count
print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Values Percentage:")
""" What percentage of data doesnt have any values"""

# do we have rows missing all values (workclass, occupation, native-country)
missing_all_three = df[
    df[["workclass", "occupation", "native-country"]].isnull().all(axis=1)
]
# print("\nRows Missing All Three (workclass, occupation, native-country):")
# print(missing_all_three)
print("\nNumber of Rows Missing All Three:", len(missing_all_three))

"""
print(df['workclass'].isnull().sum()) #number of missing values in the workclass feature
"""

print(df["workclass"])

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe())

# ---- Task 1: Identify Attribute Types ----
# Categorize each attribute into nominal, binary, ordinal, continuous, or discrete
attribute_types = {
    "Nominal": ["workclass", "occupation", "relationship", "education"],  # Fill in
    "Binary": ["sex"],  # Fill in
    "Ordinal": ["education-num"],  # Fill in
    "Continuous": [
        "fnlwgt",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ],  # Fill in
    "Discrete": ["age"],  # Fill in
}

# Print the dictionary to fill out
print("\n# Task 1: Attribute Classification")
print(attribute_types)
# print(df["occupation"].unique())

# ---- Task 2: Explore Nominal Attributes ----
# Example: Count unique values for workclass
df["workclass"].value_counts().plot(kind="bar", title="Distribution of Workclass")
plt.show()
# Explore other nominal attributes

# ---- Task 3: Explore Binary Attributes ----
# Example: Distribution of income
sns.countplot(x=df["income"])
plt.title("Income Distribution")
plt.show()
#  Explore other binary attributes (e.g., sex)

# ---- Task 4: Explore Ordinal Attributes ----
# Example: Education level distribution
sns.countplot(y=df["education"], order=df["education"].value_counts().index)
plt.title("Education Levels")
plt.show()
# Compare "education" and "education-num"

# ---- Task 5: Continuous vs. Discrete Attributes ----
# Example: Age distribution
sns.histplot(df["age"], bins=30, kde=False)
plt.title("Age Distribution")
plt.show()
#  Compare "age" (continuous) with "education-num" (discrete)

# We shall need this to be able to do some transformation
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
)

# Plot distributions for categorical variables
categorical_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "race",
    "sex",
    "native-country",
]

for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(y=df[col], order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()

# ---- Task 6: Encoding Categorical Data ----
# One-Hot Encoding: Converts categorical variables into multiple binary columns
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
one_hot_encoded = one_hot_encoder.fit_transform(df[["workclass"]].dropna())
print("One-Hot Encoding Example (Workclass):")
print(one_hot_encoded[:5])

# Label Encoding: Converts categories into integer labels
label_encoder = LabelEncoder()
df["income_encoded"] = label_encoder.fit_transform(df["income"])
print("\nLabel Encoding Example (Income):")
print(df[["income", "income_encoded"]].head())

# Explore numerical features (age, hours-per-week, capital-gain, and capital-loss) to see if there are any outlies
numerical_features = ["age", "hours-per-week", "capital-gain", "capital-loss"]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_features])
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.show()

std_scaler = StandardScaler()
df["education_num_standardized"] = std_scaler.fit_transform(df[["education-num"]])
print("\nStandardized Education Num:")
print(df[["education-num", "education_num_standardized"]].head())

# ---- Task 8: Log Transformation ----
df["capital-gain-log"] = np.log1p(df["capital-gain"])
df["capital-loss-log"] = np.log1p(df["capital-loss"])
print("\nLog Transformed Capital-Gain and Capital-Loss:")
print(
    df[["capital-gain", "capital-gain-log", "capital-loss", "capital-loss-log"]].head()
)

# ---- Task 9: Binning (Discretization) ----
age_bins = [0, 25, 45, 65, 100]
age_labels = ["Young", "Adult", "Middle-aged", "Senior"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
print("\nBinned Age Groups:")
print(df[["age", "age_group"]].head())

"""
How does this data look like?
"""
print(df.info())

"""Age is bins is still somewhat had to work with, so am removing it."""
# df.drop(columns=["age_group"], inplace=True)

"""
What I can do is use winsorization to cap the limit for age; both apply a cap to top and bottom of the distribution.
"""
from scipy.stats.mstats import winsorize

df["age"] = winsorize(
    df["age"], limits=[0.01, 0.01]
)  # Cap top & bottom 1% of age values

"""
lets what the data looks like now
"""
print("winsorize: \n", df[["age", "age_group"]].head())

"""
I still havent dealt with missing values, so lets do that now.
From earlier only 27 rows missing all data, so am likely to lose 10% of the data
if I dropped it off. But will do that for now.
"""
df.dropna(inplace=True)

# Show missing values count
print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Values Percentage:")
print((df.isnull().sum() / df.shape[0]) * 100)

plt.figure(figsize=(8, 6))
sns.kdeplot(x=df["age"], y=df["hours-per-week"], cmap="Blues", bins=10, fill=False)
plt.title("Bivariate KDE: Hours Worked vs. Age")
plt.xlabel("Age")
plt.ylabel("Hours Worked per Week")
plt.show()

"""
what is the shape of the data now?
"""
print(df.shape)


df_final = df.copy()

df_final.drop(
    columns=[
        "fnlwgt",
        "age",
        "income",
        "education-num",
        "capital-gain",
        "capital-loss",
        # "hours-per-week",
    ],
    inplace=True,
)

print("df_final: \n", df_final.info())

# sns.kdeplot(data=df_final, x="income_encoded", y="hours_per_week_scaled", fill=False)
sns.kdeplot(data=df_final, x="income_encoded", y="hours-per-week", fill=False)
plt.title("Distribution of Age and Hours per Week")

""" what is the distrbution of age"""

# Save the data frame to local.
