import numpy as np
import pandas as pd
from gradientDescend import lm


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

y_class = np.where(y>np.median(y),1,0)

y_class

predict(logit, x_norm,y_class, x_norm,y_class, verbose=False, maxiter=100000, alpha =0.01, restart=1, tol= 1e-8)

logistic(x_norm,y_class, verbose=False, maxiter=100000, alpha =0.01, restart=1, tol= 1e-8)

model = lm(verbose=False, maxiter=100000, alpha =0.01, restart=1, tol= 1e-8)

model.fit(x_norm, y)


a=[]
c=[]
for alpha in np.log10(np.logspace(0.001,0.4,200)):
    beta, mincost = lm(X=x_norm, Y=y, verbose=False, maxiter=10000, alpha =alpha, restart=1)
    a+=[alpha]
    c+=[mincost]
    print('{0:0.2f} --> {1:0.2f}'.format(alpha, mincost))



import matplotlib.pyplot as plt

plt.plot(a, c, 'o')


##################### Comparing with Statmodel #############################
df.describe()

import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

mod = sm.OLS(y, x_norm)
res = mod.fit()
res.summary()

mod1=smd.Logit(y_class, x_norm).fit()
mod1.summary()

