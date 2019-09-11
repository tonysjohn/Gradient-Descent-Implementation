import numpy as np
import pandas as pd


class lm():
    def __init__(self, alpha=0.001, verbose=True, maxiter=10000, restart=1, tol = 1e-6):
        self.X=0
        self.y=0
        self.b,self.W = (0,0)
        self.m,self.n = (0,0)
        self.alpha = alpha
        self.verbose = verbose
        self.maxiter=maxiter
        self.restart = restart
        self.tol = tol

    ################## initializing beta ####################################

    def initialize(self, X, y):
        """
        input:  n - number of features
        output: b - beta0
                W - beta
        weights assigned to near zero so that logistic regression converges faster
        """
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.b = np.random.rand()*0.01
        self.W = np.random.rand(self.n,1)*0.01

    ################## computing OLS cost and gradient ######################

    def ols(self):
        ypred, cost = self.predictlm()
        err= self.y - ypred
        db = -(1/self.m)*np.sum(err)
        dW = -(1/self.m)*self.X.T.dot(err)
        return (cost, db, dW)

    ####################### predict lm #####################################

    def predictlm(self, newX=None, newy=None):
        if newX is None:
            ypred = np.dot(self.X,self.W) + self.b
            err = (self.y - ypred)
            cost = (1/(2*self.m)) * np.sum(np.square(err))
            return (ypred, cost)
        
        m,n =newX.shape
        ypred = np.dot(newX,self.W) + self.b
        err = (newy - ypred)
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

    def method(self):
            return self.ols()
    
    def betaUpdate(self,db, dW):
        self.b = self.b - self.alpha*db
        self.W = self.W - self.alpha*dW


    ##################### gradient descent #################################

    def descend(self):
        i = 0
        cost_hist = [np.inf]
        cost, db, dW = self.method()
        cost_hist.append(cost)
        i += 1
        self.betaUpdate(db,dW)

        while (cost_hist[i-1] - cost_hist[i] > self.tol):
            cost, db, dW = self.method()
            self.betaUpdate(db,dW)
            cost_hist.append(cost)
            i += 1

            if self.verbose==True:
                print(cost_hist[i])
            
            if (i==self.maxiter):
                print("Max Threshold Exceeded")
                return cost_hist

        if cost_hist[i-1] < cost_hist[i]:
            print('Decrease alpha : solution not converging')
            return cost_hist
        if i < self.maxiter:
            print('Solution found in {}th iteration'.format(i))   
        return cost_hist


    #################### LM wrapper and random restart #######################

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1,1)
        mincost=np.inf
        for i in range(self.restart):
            self.initialize(X,y)
            cost_hist = self.descend()
            if cost_hist[-1] < mincost:
                finalb = self.b
                finalW = self.W.reshape(-1,1)
                mincost = cost_hist[-1]
            else:
                continue
        return finalb, finalW, cost_hist

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


    def predict(self, newX=None, newy=None):
        if (self.b==0 and self.W ==0):
            print("Fit the model before calling predict")
            return None
        if newX is None:
            return (self.predictlm(), None)
        else:
            newX = np.array(newX)
            newy = np.array(newy).reshape(-1,1)
            return (self.predictlm(), self.predictlm(newX,newy))
    
class logit(lm):
    def method(self):
        return self.mle()