import numpy as np

class linear_regression:
    def __init__(self, lr=0.01, epochs=20000):
         self.lr = lr
         self.epochs = epochs
         self.w = None
         self.b = None
         self.mean=None
         self.std=None
    
    def fit(self,x_train,y_train):
        self.mean=np.mean(x_train, axis=0)
        self.std=np.std(x_train, axis=0)
        x=(x_train - self.mean) / (self.std + 1e-8)
        n=x.shape[1]
        w=np.zeros(n)
        b=0
        m=len(x)

        for i in range(self.epochs):
            fw_b=np.dot(x,w)+b
            error=fw_b-y_train

            dj_dw=(1/m)*np.dot(x.T,error)
            dj_db=(1/m)*np.sum(error)

            w-=self.lr*dj_dw
            b-=self.lr*dj_db
        self.w=w
        self.b=b

    def parameters(self):
        return f"w={self.w}, b={self.b}"
    
    def predict(self,input):
        x_input=np.array(input)
        if x_input.ndim==1:
            x_input=x_input.reshape(1,-1)
        x_input=(x_input-self.mean)/(self.std+1e-8)
        out=np.dot(x_input,self.w)+self.b
        return out


        