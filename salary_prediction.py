# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Reading data from the CSV file
df = pd.read_csv("Data/survey_results_public.csv")

# Displaying the first 5 rows of the dataset
df.head()

# Selecting relevant features for analysis
df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
df = df.rename({"ConvertedComp": "Salary"}, axis=1)  # Renaming the target variable to 'Salary'
df.head()

# Dropping rows where Salary is null
df = df[df["Salary"].notnull()]
df.head()

# Checking the structure and data types
df.info()

# Dropping rows with any remaining null values
df = df.dropna()
df.isnull().sum()

# Filtering the data to include only full-time employed individuals
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)  # Dropping the Employment column since it is no longer needed
df.info()

# Exploring unique countries and handling them
df['Country'].value_counts()

# Function to group smaller categories as 'Other' based on a cutoff frequency
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# Mapping countries with fewer entries to 'Other'
country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()

# Visualizing salary distribution across countries
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# Removing outliers from the dataset by applying salary filters
df = df[df["Salary"] <= 250000]
df = df[df["Salary"] >= 10000]
df = df[df['Country'] != 'Other']

# Replotting the boxplot after outlier removal
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# Checking unique values in 'YearsCodePro'
df["YearsCodePro"].unique()

# Cleaning the 'YearsCodePro' column by converting text to numeric values
def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
df["YearsCodePro"].unique()

# Checking unique values in 'EdLevel'
df["EdLevel"].unique()

# Simplifying education levels by grouping similar categories
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

df['EdLevel'] = df['EdLevel'].apply(clean_education)
df["EdLevel"].unique()

# Label encoding categorical features for modeling
from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()

le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df["Country"].unique()

# Splitting the dataset into features (X) and target (y)
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Training a Linear Regression model
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)

# Predicting with the trained Linear Regression model
y_pred = linear_reg.predict(X)

# Evaluating model performance using Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))
error

# Training a Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)

# Predicting with the Decision Tree Regressor
y_pred = dec_tree_reg.predict(X)

# Evaluating Decision Tree performance
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# Training a Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(X, y.values)

# Predicting with the Random Forest Regressor
y_pred = random_forest_reg.predict(X)

# Evaluating Random Forest performance
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# Hyperparameter tuning for Decision Tree using GridSearchCV
from sklearn.model_selection import GridSearchCV

max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

# Fitting the best estimator from GridSearchCV
regressor = gs.best_estimator_
regressor.fit(X, y.values)

# Predicting with the tuned Decision Tree Regressor
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

# Testing the model on new data
X = np.array([["United States", 'Master’s degree', 15 ]])
X

# Transforming categorical data for prediction
X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X

# Predicting salary for the new data input
y_pred = regressor.predict(X)
y_pred

# Saving the trained model and label encoders using pickle
import pickle
data = {"model": regressor, "le_country": le_country, "le_education": le_education}
with open('saved_steps.pkl', 'wb') as file:
    pickle.dump(data, file)

# Loading the saved model and label encoders for future predictions
with open('saved_steps.pkl', 'rb') as file:
    data = pickle.load(file)

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Predicting with the loaded model
y_pred = regressor_loaded.predict(X)
y_pred
