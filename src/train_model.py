import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load data
dataset_path = "../ml-100k/u.data"
df = pd.read_csv(dataset_path, sep= "\t", names=["user_id", "item_id", "rating","timestamp"] )

#select features and targets
x = df[["user_id", "item_id"]]
y = df["rating"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#create model
model = LinearRegression()

#train model
model.fit(X_train,  y_train)

#predict on the test set
y_pred = model.predict(X_test)

#Evaluate Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")