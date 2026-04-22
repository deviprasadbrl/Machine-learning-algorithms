from algo import logistic_regression
import pandas as pd

df=pd.read_csv('Cancer_Data.csv')      
modified_df=df.drop("Unnamed: 32",axis=1)
x = modified_df.select_dtypes(include=["float64"]).values
df["diagnosis"]=df["diagnosis"].map({"M":1,"B":0})
y=df["diagnosis"].values

model=logistic_regression()

model.fit(x,y)
print(model.parameters())
test=[13.9,19.24,88.73,602.9,0.07991,0.05326,0.02995,0.0207,0.1579,0.05594,0.3316,0.9264,2.056,28.41,0.003704,0.01082,0.0153,0.006275,0.01062,0.002217,16.41,26.42,104.4,830.5,0.1064,0.1415,0.1673,0.0815,0.2356,0.07603]
print(model.predict(test))
