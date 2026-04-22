import numpy as np


class logistic_regression:
    def __init__(self,lr=0.01,epochs=10000):
        self.lr=lr
        self.epochs=epochs
        self.w = None
        self.b = None
        self.mean=None
        self.std=None


    def sigmoid(self,z):
        z = np.clip(z,-500,500)
        return 1/(1+np.exp(-z))
    

    def fit(self,x,y):
        self.mean=np.mean(x,axis=0)
        self.std=np.std(x,axis=0)
        x_s=(x-self.mean)/(self.std+1e-8)
        m,n=x_s.shape
        w=np.zeros(n)
        b=0
        for i in range(self.epochs):
            fw_b=self.sigmoid(np.dot(x_s,w)+b)
            eror=fw_b-y
            dj_dw=(1/m)*np.dot(x_s.T,eror)
            dj_db=(1/m)*np.sum(eror)

            w-=self.lr*dj_dw
            b-=self.lr*dj_db
        
        self.w=w
        self.b=b

    def predict(self,input):
        x_input = np.array(input)
        if x_input.ndim == 1:
            x_input = x_input.reshape(1, -1)
        x_input=(x_input-self.mean)/(self.std+1e-8)
        out=self.sigmoid(np.dot(x_input,self.w)+self.b)
        pred=np.where(out>=0.5,1,0)
        return "M" if pred==1 else "B"
    
    def parameters(self):
        return f"w={self.w},b={self.b}"
    
