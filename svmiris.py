from sklearn import datasets
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() 

x = iris.get('data')
y = iris.get('target')
num = x.shape[0]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=87)
num_test = x_test.shape[0] 
clf_rbf = svm.SVC(decision_function_shape="ovo", kernel="rbf")
clf_rbf.fit(x_train, y_train)
y_test_pre_rbf = clf_rbf.predict(x_test)
print(clf_rbf.score(x_test,y_test))