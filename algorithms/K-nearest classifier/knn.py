from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
iris = datasets.load_iris()                # Loading iris dataset

# Printing description
features = iris.data
labels = iris.target
n_feature = iris.data[:,np.newaxis,2]
#print(features[0],labels[0])
#print(iris.DESCR)

# Training classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

# Prediction 
prediction = clf.predict([[12,1,1,1]])
if prediction == [0]:
    output = 'satosa'
elif prediction == [2]:
    output = 'verginica'
else :
    output = 'Versicolour'
print(output)

plt.scatter(n_feature, labels)
plt.show()