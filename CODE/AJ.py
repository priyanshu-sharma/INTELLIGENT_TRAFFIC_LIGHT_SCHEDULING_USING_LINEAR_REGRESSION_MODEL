#Data handling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

#Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

#Data Preprocessing
data=pd.read_csv("data.csv")
data.columns=['time','North','East','West','South','NGR','EGR','WGR','SGR']
data['time']=data.index%96


#Print data head and desciption
data.head()

#Print Pairgrid
sb.set(style='whitegrid',context='notebook')
cols=['time','North','East','West','South']
sb.pairplot(data[cols],size=3)
plt.show()

#Print Corrcoef grid
cm = np.corrcoef(data[cols].values.T)
sb.set(font_scale=1.5)
hm=sb.heatmap(cm,
               cbar=True,
               annot=True,
               square=True,
               fmt='.2f',
               annot_kws={'size':15},
               yticklabels=cols,
               xticklabels=cols)
plt.show()


#Spliting data for training and testing
X=data[cols].values
y=data[['NGR','EGR','WGR','SGR']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.25)

#Linear Regression Model
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)


#Predict green ratios args('time','North','East','West','South')
lr.predict([31,162,157,104,114])
