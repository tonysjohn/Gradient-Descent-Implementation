import numpy as np
import pandas as pd

################## initializing beta ####################################

def initbeta(n):
    """
    input:  n - number of features
    output: b - beta0
            W - beta
    weights assigned to near zero so that logistic regression converges faster
    """
    b = np.random.rand()*0.01
    W = np.random.rand(n,1)*0.01
    return (b,W)

################## computing OLS cost and gradient ######################

def ols(X,y,b=0,W=0):
    m,n = X.shape
    ypred, cost = predictlm(X,y,b,W)
    err= y - ypred
    db = -(1/m)*np.sum(err)
    dW = -(1/m)*X.T.dot(err)
    return (cost, db, dW)

####################### predict lm #####################################

def predictlm(X,y,b,W):
    m,n = X.shape
    ypred = np.dot(X,W) + b
    err = (y - ypred)
    cost = (1/(2*m)) * np.sum(np.square(err))
    return (ypred, cost)

################## computing MLE cost and gradient ######################

def mle(X,y,b=0,W=0):
    m,n = X.shape
    ypred, cost = predictlogit(X,y,b,W)
    err= y - ypred
    db = -np.sum(err)/float(m)
    dW = -(1/m)*X.T.dot(err)
    return (cost, db, dW)

####################### predict logit ##################################

def predictlogit(X,y,b,W):
    m,n = X.shape
    z = np.dot(X,W) + b
    ypred = 1/(1+np.exp(-z))
    err = (y - ypred)
    cost = -(1/m) * (y.T.dot(np.log(ypred)).reshape(-1)+(1-y).T.dot(np.log(1-ypred)).reshape(-1))
    return (ypred, cost)

######################### selecting method #############################

def method(X,y,b,W,mth):
    if mth == "lm":
        return ols(X,y,b,W)
    elif mth == "binary":
        return mle(X,y,b,W)


##################### gradient descent #################################

def descend(X, y, mth, alpha=0.001, verbose=True, maxiter=10000, tol=1e-6):
    m,n = X.shape
    b,W = initbeta(n)
    oldcost,i = np.inf,0
    cost, db, dW = method(X,y,b,W,mth)
    b = b - alpha*db
    W = W - alpha*dW
    while (oldcost - cost > tol):
        oldcost = cost
        cost, db, dW = method(X,y,b,W,mth)

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
    if i < maxiter:
        print('Solution found in {}th iteration'.format(i))   
    return (b,W.reshape(-1,1),cost)


#################### LM wrapper and random restart #######################

def lm(X, y, alpha=0.001, verbose=True, maxiter=10000, restart=1, tol = 1e-6):
    mincost=np.inf
    for i in range(restart):
        b,W, cost = descend(X, y, mth="lm", alpha=alpha, verbose=verbose,maxiter=maxiter, tol=tol)
        if cost < mincost:
            finalb = b
            finalW = W
            mincost = cost
        else:
            continue
    return finalb, finalW, mincost

#################### Logistic wrapper and random restart #######################

def logit(X, y, alpha=0.001, verbose=True, maxiter=10000, restart=1, tol = 1e-6):
    mincost=np.inf
    for i in range(restart):
        b,W, cost = descend(X, y, mth="binary", alpha=alpha, verbose=verbose,maxiter=maxiter, tol=tol)
        if cost < mincost:
            finalb = b
            finalW = W
            mincost = cost
        else:
            continue
    return finalb, finalW, mincost

########################### Predict function ################################


def predict(method, X, y, newX=None, newy=None, alpha=0.001, verbose=True, maxiter=10000, restart=1, tol = 1e-6):
    if method not in [lm,logit]:
        print("Please provide a method")
        return None
    b, W, cost = method(X, y, alpha=alpha, verbose=verbose, maxiter=maxiter, restart=restart, tol = tol)
    if newX is None:
        if method == lm:
            return (predictlm(X,y,b,W), (None,None))
        elif method == logit:
            return (predictlogit(X,y,b,W), (None,None))
    else:
        if method == lm:
            return (predictlm(X,y,b,W), predictlm(newX,newy,b,W))
        elif method == logit:
            return (predictlogit(X,y,b,W), predictlogit(newX,newy,b,W))

