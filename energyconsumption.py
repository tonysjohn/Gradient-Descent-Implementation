import numpy as np
import pandas as pd
from gradientDescend_oops import lm


###################### reading data #######################################
df=pd.read_csv(r".\energydata_complete.csv")
df.date=pd.to_datetime(df.date)
y=df["Appliances"]
x=df[['T1','T2','T3','T4','T5','T6']]
x=np.array(x)
x_norm=(x-x.mean(axis=0))/x.std(axis=0)
x_norm.shape
y
y=np.array(y).reshape(-1,1)


np.random.seed(100)
split = np.random.uniform(size=x.shape[0]) < 0.7

X_train= x[split]
y_train= y[split]
X_test= x[~split]
y_test= y[~split]
x_train_mean = X_train.mean(axis=0)
x_train_std = X_train.std(axis=0)
X_train_norm = (X_train-x_train_mean)/x_train_std
X_test_norm = (X_test-x_train_mean)/x_train_std

y_class = np.where(y>np.median(y),1,0)

(1/(2*y_test.shape[0])) * np.sum(np.square(y_test-y.mean()))

y_class

predict(logit, x_norm,y_class, x_norm,y_class, verbose=False, maxiter=100000, alpha =0.01, restart=1, tol= 1e-8)

logistic(x_norm,y_class, verbose=False, maxiter=100000, alpha =0.001, restart=1, tol= 1e-8)

model = lm(verbose=False, maxiter=100000, alpha =0.1, restart=1, tol= 1e-8)

finalb, finalW, cost_hist = model.fit(X_train_norm, y_train)


model.predict(X_test_norm,y_test)

cost_hist[-1]


a=[]
c=[]
for alpha in np.log10(np.logspace(0.001,0.4,200)):
    beta, mincost = lm(X=x_norm, Y=y, verbose=False, maxiter=10000, alpha =alpha, restart=1)
    a+=[alpha]
    c+=[mincost]
    print('{0:0.2f} --> {1:0.2f}'.format(alpha, mincost))



import matplotlib.pyplot as plt

plt.plot(cost_hist)

plt.plot(a, c, 'o')


##################### Comparing with Statmodel #############################
df.describe()

import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

mod = sm.OLS(y_train, X_train_norm)
res = mod.fit()
res.summary()

mod1=smd.Logit(y_class, x_norm).fit()
mod1.summary()

