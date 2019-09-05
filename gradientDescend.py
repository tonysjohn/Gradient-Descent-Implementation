import numpy as np
import pandas as pd

################## initializing beta ##########################

def initbeta(n):
    b = np.random.rand()
    W = np.random.rand(n,1)
    return (b,W)

################## computing cost and gradient ##########################

def rmse(X,y,b=0,W=0):
    m,n = x.shape
    ypred = np.dot(X,W) + b
    err = (y - ypred)
    #cost = 1/(2*m)*(err.dot(err.T))
    cost = (1/(2*m)) * np.sum(np.square(err))
    db = -np.sum(err)/float(m)
    #dW = 1/m*((err*x).sum(axis=1)).reshape(-1,1)
    #print(err.shape,X.shape)
    dW = -(1/m)*X.T.dot(err)
    #dW = -(1/m)*np.sum(np.multiply(X,err), axis=0)
    return (cost, db, dW)


##################### gradient descent ######################

def descend(x, y, alpha=0.001, verbose=True, maxiter=100, tol=1e-6):
    m,n = x.shape
    b,W = initbeta(n)
    oldcost,i = np.inf,0
    cost, db, dW = rmse(x,y,b,W)
    b = b - alpha*db
    W = W - alpha*dW
    while (oldcost - cost > tol):
        oldcost = cost
        cost, db, dW = rmse(x,y,b,W)

        b = b- alpha*db
        W = W- alpha*dW

        if verbose==True:
            print(cost)
        i+=1
        
        if (i==maxiter):
            print("Max Threshold Exceeded")
            return (np.append(b,W),cost)
    if oldcost < cost:
        print('Decrease alpha : solution not converging')
        return (np.nan,cost)
    elif i < maxiter:
        print('Solution found in {}th iteration'.format(i))   
    return (np.append(b,W),cost)

#################### LM wrapper and random restart ##############

def lm(X, Y, alpha=0.001, verbose=True, maxiter=100, restart=1, tol = 1e-6):
    mincost=np.inf
    
    for i in range(restart):
        beta, cost = descend(x=X, y=Y, alpha=alpha, verbose=verbose,maxiter=maxiter, tol=tol)
        if cost < mincost:
            #print('{0:0.2f} at {1}th iteration'.format(cost,i+1))
            finalbeta = beta
            mincost = cost
        else:
            continue
    return finalbeta, mincost

################### testing data ##############################

# x = np.arange(10).reshape(1,-1)
# z = np.arange(100,110)
# y= 2*x+z+50
# y=2*x+100
# x=np.array([x,z])

###################### reading data ###########################
df=pd.read_csv(r".\energydata_complete.csv")
df.date=pd.to_datetime(df.date)
y=df["Appliances"]
x=df[['T1','T2','T3','T4','T5','T6']]
x=np.array(x)
x_norm=(x-x.mean(axis=0))/x.std(axis=0)
x_norm
y=np.array(y).reshape(-1,1)


lm(X=x_norm, Y=y, verbose=False, maxiter=100000, alpha =0.1, restart=1, tol= 1e-8)

a=[]
c=[]
for alpha in np.log10(np.logspace(0.001,0.4,200)):
    beta, mincost = lm(X=x_norm, Y=y, verbose=False, maxiter=10000, alpha =alpha, restart=1)
    a+=[alpha]
    c+=[mincost]
    print('{0:0.2f} --> {1:0.2f}'.format(alpha, mincost))



import matplotlib.pyplot as plt

plt.plot(a, c, 'o')


##################### Comparing with Statmodel ################
df.describe()

import statsmodels.api as sm

mod = sm.OLS(y, x_norm)
res = mod.fit()
res.summary()

