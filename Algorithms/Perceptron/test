from algo import Perceptron
import pandas as pd

df=pd.read_csv(r'C:\Users\Devi Prasad\OneDrive\Documents\prep\Machine learning\DEEP LEARNING\perceptron\placement.csv')

df["cgpa"] = df["cgpa"].fillna(df["cgpa"].mean())
df["iq"] = df["iq"].fillna(df["iq"].mean())

x=df[["cgpa","iq"]].values
y=df["placement"].values

model=Perceptron()
model.fit(x,y)
print(model.parameters())
test=[8.1,166]
print(model.predict(test))