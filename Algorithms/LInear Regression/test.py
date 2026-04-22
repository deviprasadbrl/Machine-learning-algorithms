from algo import linear_regression
import pandas as pd

df=pd.read_csv(r'insurance.csv')

df["smoker"] = df["smoker"].map({'yes': 1, 'no': 0})

x=df[["age","bmi","children","smoker"]].values.astype(float)
y=df["charges"].values.astype(float)


model=linear_regression()
model.fit(x,y)
print(model.parameters())
print(model.predict([26,20.8,0,0]))


