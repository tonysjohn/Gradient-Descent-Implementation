import numpy as np
import pandas as pd

# reading data 
df=pd.read_csv(r".\energydata_complete.csv")
df.date=pd.to_datetime(df.date)

y=df["Appliances"]

x=df[['T1','T2']]

x=np.array(x).T
x_norm=(x-x.mean())/x.std()
x_norm
y=np.array(y)

x = np.arange(10).reshape(1,10)
y= x+100


def initbeta(n):
    b = np.random.rand()
    W = np.random.rand(n,1)
    return (b,W)

initbeta(5)

def rmse(n,m,err,X):
    #cost = 1/(2*m)*(err.dot(err.T))
    cost = np.sum(err**2)/float(2*m)
    db = -np.sum(err)/float(m)
    #dW = 1/m*((err*x).sum(axis=1)).reshape(-1,1)
    #print(err.shape,X.shape)
    dW = -X.dot(err.T).reshape(-1,1)/float(m)
    return (cost, db, dW)



def descend(x, y, alpha=0.001, verbose=True, maxiter=100):
    n,m = x.shape
    b,W = initbeta(n)
    oldcost,i = np.inf,0
    ypred = (W.T).dot(x) + b
    err = (y - ypred).reshape(-1)
    cost, db, dW = rmse(n,m,err,x)
    b = b - alpha*db
    W = W - alpha*dW
    while (oldcost- cost>1e-6):
        oldcost = cost

        ypred = (W.T).dot(x) + b

        #error
        err = (y - ypred).reshape(-1)

        #update
        cost, db, dW = rmse(n,m,err,x)

        b = b- alpha*db
        #print(dW.shape)
        #print(W.shape)
        W = W- alpha*dW

        if verbose==True:
            print(cost)
        i+=1
        
        if (i==maxiter):
            break
        #print(ypred)   
    return (b,W,cost)

def lm(X, Y, alpha=0.001, verbose=True, maxiter=100, restart=500):
    mincost=np.inf
    
    for i in range(restart):
        b, W, cost = descend(x=X, y=Y, alpha=alpha, verbose=verbose,maxiter=maxiter)
        if cost < mincost:
            print('{0:0.2f} at {1}th iteration'.format(cost,i+1))
            finalb = b
            finalW = W
            mincost = cost
        else:
            continue
    return finalb,finalW, mincost

lm(X=x, Y=y, verbose=False, maxiter=50000, alpha =0.0001, restart=1)

