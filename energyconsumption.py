import numpy as np
import pandas as pd
from gradientDescend import *


###################### reading data #######################################
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


##################### Comparing with Statmodel #############################
df.describe()

import statsmodels.api as sm

mod = sm.OLS(y, x_norm)
res = mod.fit()
res.summary()

