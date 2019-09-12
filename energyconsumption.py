import numpy as np
import pandas as pd
from gradientDescend_oops import lm,logit
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
#from gradientDescend import logit

###################### reading data #######################################
df=pd.read_csv(r".\energydata_complete.csv")
df.date=pd.to_datetime(df.date)
y=df["Appliances"]
x=df[['T1','T2','T3','T4','T5','T6']]


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

y_train_class = np.where(y_train>np.median(y_train),1,0)
y_test_class = np.where(y_test>np.median(y_train),1,0)

train_hist = []
test_hist = []
cost_iter_hist = []
alphas = 10**(np.arange(-5,0,0.5))
for alpha in alphas:
    model = lm(verbose=False, maxiter=100000, alpha =alpha, restart=1, tol= 1e-8)
    finalb, finalW, cost_hist = model.fit(X_train_norm, np.log(y_train))
    (_,train_cost),(_,test_cost) = model.predict(X_test_norm,np.log(y_test))
    cost_iter_hist += [cost_hist]
    train_hist += [train_cost]
    test_hist += [test_cost]

plt.plot(np.log10(alphas), train_hist)
plt.plot(np.log10(alphas), test_hist)

for i in cost_iter_hist:
    plt.plot(len(i))




train_hist = []
test_hist = []
tols = 10**(np.arange(-12,2,0.5))#[1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]
for tol in tols:
    model = lm(verbose=False, maxiter=100000, alpha =0.001, restart=1, tol= tol)
    finalb, finalW, cost_hist = model.fit(X_train_norm, np.log(y_train))
    (_,train_cost),(_,test_cost) = model.predict(X_test_norm,np.log(y_test))
    train_hist += [train_cost]
    test_hist += [test_cost]

plt.plot(np.log10(tols), train_hist)
plt.plot(np.log10(tols), test_hist)




model = logit(verbose=False, maxiter=100000, alpha =0.q, restart=1, tol= 1e-8)
finalb, finalW, cost_hist = model.fit(X_train_norm, y_train_class)
model.predict(X_test_norm,y_test_class)


cost_hist[-1]




a=[]
c=[]
for alpha in np.log10(np.logspace(10,1,10)):
    beta, mincost = lm(X=x_norm, Y=y, verbose=False, maxiter=10000, alpha =alpha, restart=1)
    a+=[alpha]
    c+=[mincost]
    print('{0:0.2f} --> {1:0.2f}'.format(alpha, mincost))

plt.plot(cost_hist)

plt.plot(a, c, 'o')


##################### Comparing with Statmodel #############################
df.describe()

import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

mod = sm.OLS(np.log(y_train), X_train_norm)
res = mod.fit()
res.summary()

mod1=smd.Logit(y_train_class, X_train_norm).fit()
mod1.summary()

import seaborn as sns
sns.distplot(np.log(y))
