Linear regression :
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.datasets import load_iris
 iris_df = pd.read_csv('iris.data.csv')
 iris_df.head()

 data = load_iris()
 data.feature_names

 data.target_names
  data.target
  
   X = data.data
 X.shape
 
  y=data.target
 y.shape

  y = y.reshape(-1, 1)
 y.shape

 plt.figure(figsize=(18,8),dpi=100)
 plt.scatter(X.T[1],X.T[2])
 plt.title('IRIS Petal and sepal length', fontsize=20)
 plt.ylabel('Petal Length')
 plt.xlabel('sepal length')

  from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LinearRegression
 X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20)
 lr = LinearRegression()
 iris_model = lr.fit(X_train, y_train)
 predictions = iris_model.predict(X_test)
 from sklearn.metrics import r2_score   #class will help us to calculate and see the score of our predictions
 r2_score(y_test, predictions)

  np.sqrt(((predictions - y_test)**2).mean())


  SVM :


   from sklearn import datasets
 cancer_data = datasets.load_breast_cancer()
 print(cancer_data.data[5])

 print(cancer_data.data.shape)
 #target set
 print(cancer_data.target)

  from sklearn.model_selection import train_test_split
 cancer_data = datasets.load_breast_cancer()
 X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4,random_state=109)
 from sklearn import svm
 #create a classifier
 cls = svm.SVC(kernel="linear")
 #train the model
 cls.fit(X_train,y_train)
 #predict the response
 pred = cls.predict(X_test)

 from sklearn import metrics
 #accuracy
 print("acuracy:", metrics.accuracy_score(y_test,y_pred=pred))
 #precision score
 print("precision:", metrics.precision_score(y_test,y_pred=pred))
 #recall score
 print("recall" , metrics.recall_score(y_test,y_pred=pred))
 print(metrics.classification_report(y_test, y_pred=pred))


  import matplotlib.pyplot as plt
 from sklearn import datasets
 from sklearn import svm
 #loading the dataset
 letters = datasets.load_digits()
 #generating the classifier
 clf = svm.SVC(gamma=0.001, C=100)
 #training the classifier
 X,y = letters.data[:-10], letters.target[:-10]
 clf.fit(X,y)
 #predicting the output
 print(clf.predict(letters.data[:-10]))
 plt.imshow(letters.images[6], interpolation='nearest')
 plt.show()
