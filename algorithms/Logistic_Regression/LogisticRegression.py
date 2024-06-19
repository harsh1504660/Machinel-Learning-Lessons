#Train a logistic regresser mode to predit wheter flower is verginica or not
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()

#print(list(iris.keys()))
#print(iris.data)
#print(iris.target)
#print(iris.DESCR)

x = iris['data'][:,3:]
y = (iris['target'] == 2).astype(int)
#print(x)
#print(y)

#Training 
clf = LogisticRegression()
clf.fit(x,y)

example = clf.predict([[12.6]])
print(example)

#Using matplotib to plot the visulaizaton
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob)
plt.show()