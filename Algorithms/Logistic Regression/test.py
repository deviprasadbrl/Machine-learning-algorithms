from algo import logistic_regression
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv('Cancer_Data.csv')      
modified_df=df.drop("Unnamed: 32",axis=1)
x = modified_df.select_dtypes(include=["float64"]).values
df["diagnosis"]=df["diagnosis"].map({"M":1,"B":0})
y=df["diagnosis"].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=logistic_regression(x_train,y_train,x_test,y_test)

model.fit()
print(model.parameters())
model.scale_data()
test=[17.05,19.08,113.4,895,0.1141,0.1572,0.191,0.109,0.2131,0.06325,0.2959,0.679,2.153,31.98,0.005532,0.02008,0.03055,0.01384,0.01177,0.002336,19.59,24.89,133.5,1189,0.1703,0.3934,0.5018,0.2543,0.3109,0.09061]
print(model.predict(test))