import numpy as np
import pandas as pd

################## initializing beta ####################################

def initbeta(n):
    b = np.random.rand()
    W = np.random.rand(n,1)
    return (b,W)

################## computing cost and gradient ##########################

def rmse(X,y,b=0,W=0):
    m,n = X.shape
    ypred = np.dot(X,W) + b
    err = (y - ypred)
    # cost = 1/(2*m)*(err.dot(err.T))
    cost = (1/(2*m)) * np.sum(np.square(err))
    db = -np.sum(err)/float(m)
    # dW = 1/m*((err*x).sum(axis=1)).reshape(-1,1)
    # print(err.shape,X.shape)
    dW = -(1/m)*X.T.dot(err)
    # dW = -(1/m)*np.sum(np.multiply(X,err), axis=0)
    return (cost, db, dW)


##################### gradient descent #################################

def descend(X, y, alpha=0.001, verbose=True, maxiter=100, tol=1e-6):
    m,n = X.shape
    b,W = initbeta(n)
    oldcost,i = np.inf,0
    cost, db, dW = rmse(X,y,b,W)
    b = b - alpha*db
    W = W - alpha*dW
    while (oldcost - cost > tol):
        oldcost = cost
        cost, db, dW = rmse(X,y,b,W)

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


#################### LM wrapper and random restart #######################

def lm(X, y, alpha=0.001, verbose=True, maxiter=100, restart=1, tol = 1e-6):
    mincost=np.inf
    
    for i in range(restart):
        beta, cost = descend(X, y, alpha=alpha, verbose=verbose,maxiter=maxiter, tol=tol)
        if cost < mincost:
            # print('{0:0.2f} at {1}th iteration'.format(cost,i+1))
            finalbeta = beta
            mincost = cost
        else:
            continue
    return finalbeta, mincost
