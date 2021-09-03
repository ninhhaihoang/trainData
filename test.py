import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Read input
pd.read_csv('Data.csv', header=0, skiprows=3).T.to_csv('output.csv', header = False, index=True)
df = pd.read_csv('output.csv')
print("output.csv :")
print(df)
print("---------------------------------------------------------------------------------------------------------------------")
df_raw = df.copy()


# Describe
print("Describe:")
print(df.describe())
print("---------------------------------------------------------------------------------------------------------------------")


# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# ---------------------------------------------------------------------------------------------------------------------------- #
# CLEAN DATA

# Analyse missing data
print("Missing data table:")
print(draw_missing_data_table(df))
print("---------------------------------------------------------------------------------------------------------------------")


# Update output.csv
update_df = df.drop([df.index[0], df.index[1], df.index[2], df.index[3], df.index[4], df.index[5], df.index[6], df.index[7]
, df.index[8], df.index[9], df.index[10], df.index[11], df.index[12], df.index[13], df.index[14], df.index[15], df.index[16]
, df.index[17], df.index[18], df.index[19], df.index[20], df.index[21], df.index[22], df.index[23], df.index[24], df.index[25]
, df.index[26], df.index[27], df.index[28], df.index[64]])
print("Update output.csv:")
print(update_df)
print("---------------------------------------------------------------------------------------------------------------------")


# Data type of update output
print("Data type of update output.csv")
print(update_df.dtypes)
print("---------------------------------------------------------------------------------------------------------------------")


# Data type of update later
update_df = update_df.astype(float)
print("Data type of update later:")
print(update_df.dtypes)
print("---------------------------------------------------------------------------------------------------------------------")


y = update_df(colums=('Vietnam','"South Korea"')).values



update_df = update_df.dropna(axis=1)

print(update_df)

x = update_df.drop(['Vietnam'],axis=1).values

# y = update_df['Vietnam'].values


# update_df = update_df.dropna(axis=1)

# print(update_df)

# x = update_df.drop(['Vietnam'],axis=1).values
print("Array without Viet Nam:")
print(x)
print("---------------------------------------------------------------------------------------------------------------------")

print("Array Viet Nam")
print(y)
print("---------------------------------------------------------------------------------------------------------------------")
# ---------------------------------------------------------------------------------------------------------------------------- #
# TRAINING PART

# Split the dataset in training set and test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, shuffle=False)
print("x train: ",x_train)
print("y train: ",y_train)
print("x test: ",x_test)
print("y test: ", y_test)

# Train model on the training set
ml = LinearRegression()
ml.fit(x_train,y_train)

regr = make_pipeline(StandardScaler(), SVR(C=1))
regr.fit(x_train,y_train)


# Predict the test set results
print("predict:")
y_pred = ml.predict(x_test)
print("linear regression: ",y_pred)

y_pred2 = regr.predict(x_test)
print("Epsilon-Support Vector Regression: ", y_pred2)


# Evaluate the model
print("R2 score:")
print(r2_score(y_test, y_pred))
print(r2_score(y_test, y_pred2))


# Predict value
pred_y_df = pd.DataFrame({'Actual Value':y_test,'Predict Value':y_pred,'Difference':y_test-y_pred})
print(pred_y_df[0:20])
print(MAE(y_test, y_pred))

pred_y_df2 = pd.DataFrame({'Actual Value':y_test,'Predict Value':y_pred2,'Difference':y_test-y_pred2})
print(pred_y_df2[0:20])
print(MAE(y_test, y_pred2))


# Plot the results
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")