import numpy as np

class Perceptron():
    def __init__(self,lr=0.01,epochs=10000):
        self.lr=lr
        self.epochs=epochs
        self.w=None
        self.b=None
        self.mean=None
        self.std=None

    def activation(self,z):
        return np.where(z>=0,1,0)
    
    
    def fit(self,x,y):
     self.mean=np.mean(x,axis=0)
     self.std=np.std(x,axis=0)
     x=(x-self.mean)/(self.std+1e-8)
     m,n=x.shape
     w=np.zeros(n)
     b=0

     for i in range(self.epochs):
        pred=self.activation(np.dot(x,w)+b)
        w+=self.lr*np.dot(x.T,(y-pred))
        b+=self.lr*np.sum(y-pred)
     self.w=w
     self.b=b

    def parameters(self):
       return f"w={self.w},b={self.b}"
    
    def predict(self,input):
       x_input=np.array(input)
       if x_input.ndim==1:
          x_input=x_input.reshape(1,-1)
       x_input=(x_input-self.mean)/(self.std+1e-8)
       pred=self.activation(np.dot(x_input,self.w)+self.b)
       return pred
    



        