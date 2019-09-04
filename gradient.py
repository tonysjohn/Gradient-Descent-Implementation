import numpy as np
import pandas as pd

df= pd.read_csv(r".\energydata_complete.csv")

df.head()

for col in df.columns:
    print(col, df[col].dtype)

df.date=pd.to_datetime(df.date)

y=df["Appliances"]

x=df[['T1','T2']]

x=np.array(x).T
y=np.array(y)


def initialize(n):
    beta = np.random.rand(n).reshape(-1,1)
    return beta

inter=np.ones(x.shape[1])
x=np.vstack((inter,x))


beta=initialize(x.shape[0])

for 
ypred = beta.T.dot(x)[0]


err=(ypred-y)
loss=1/(2*x.shape[1])*err.T.dot(err)
update=(1/(x.shape[1])*err*x).sum(axis=1)

beta =beta-0.01*update.reshape(-1,1)

loss


ypred

loss
dbeta=1

beta
err*x
x