import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class logistic_regression:
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        self.w = None
        self.b = None
        self.scaler=StandardScaler()


    def sigmoid(self,z):
        z = np.clip(z,-500,500)
        return 1/(1+np.exp(-z))
    
    def scale_data(self):
        self.x_train=self.scaler.fit_transform(self.x_train)
        self.x_test=self.scaler.transform(self.x_test)


    def fit(self):
        w=np.zeros(self.x_train.shape[1])
        b=0
        for i in range(10000):
            fw_b=self.sigmoid(np.dot(self.x_train,w)+b)
            eror=fw_b-self.y_train
            dj_dw=(1/len(self.x_train)*np.dot(self.x_train.T,eror))
            dj_db=(1/len(self.x_train)*np.sum(eror))

            w-=0.01*dj_dw
            b-=0.01*dj_db
        
        self.w=w
        self.b=b

    def predict(self,input):
        x_input = np.array(input)
        if x_input.ndim == 1:
            x_input = x_input.reshape(1, -1)
        x_input=self.scaler.transform(x_input)
        out=self.sigmoid(np.dot(x_input,self.w)+self.b)
        pred=np.where(out>=0.5,1,0)
        return "M" if 1 else "B"
    
    def interpred(self):
        return f"The most wieghted weight:{max(self.w)}"
    
    def parameters(self):
        return f"w={self.w},b={self.b}"
    