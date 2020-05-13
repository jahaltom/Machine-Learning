import csv 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

##5A###Plot the Training set
X_Train = []  
Y_Train = []       
with open('HW2train.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
      X_Train.append(float(row[1]))
      Y_Train.append(int(row[0]))

plt.plot(X_Train, Y_Train, '.' 'k' )
plt.show()
##5B##Fit a logistic model to predict y.
X_Train = np.array(X_Train)
X_Train = X_Train.reshape(-1,1)
LR= LogisticRegression(fit_intercept=True,penalty = 'none', solver = 'lbfgs')
LR.fit(X_Train,Y_Train)
#print intercept, coefficients, and accuracy
print(LR.score(X_Train,Y_Train))
print(LR.coef_)
print(LR.intercept_)
#####Plot Prob(y = 1|x) 
#Generate uniformly spaced values along the horizontal axis
Uni = []
Uni = np.linspace(0, 100, num=1000)
Uni = Uni.reshape(-1,1)
PY1X = (LR.predict_proba(Uni))[:, 1]##Get 2nd column Prob(y = 1|x) 
plt.title('HW2train Scatter Plot and Prob(y = 1|x)')
plt.plot(Uni, PY1X, '.' 'r')
plt.plot(X_Train, Y_Train, '.' 'k' )
plt.show()

##Plot the Test data 
X_Test = []  
Y_Test = []       
with open('HW2test.csv', 'r') as rf:
    reader = csv.reader(rf, delimiter=',')
    for row in reader:
      X_Test.append(float(row[1]))
      Y_Test.append(int(row[0]))
plt.title('HW2test Scatter Plot and Prob(y = 1|x)')
plt.plot(Uni, PY1X, '.' 'r' )
plt.plot(X_Test, Y_Test, '.' 'k')
X_Test = np.array(X_Test)
X_Test = X_Test.reshape(-1,1)
print(LR.score(X_Test,Y_Test))##Test accuracy
plt.show()

##5C####K-Nearest Neighbors

neigh = KNeighborsClassifier(n_neighbors=1, weights='uniform',algorithm='auto')
neigh.fit(X_Train, Y_Train)
print("Training Accuracy",neigh.score(X_Train,Y_Train))
Yx = (neigh.predict(Uni))
plt.plot(Uni, Yx, 'r')
plt.plot(X_Train, Y_Train, '.' 'k' )
plt.title('1nn Classifier with Training data’ for k = 1')
plt.show()
print("Testing Accuracy",neigh.score(X_Test,Y_Test))
plt.title('1nn Classifier with Testing data’ for k = 1')
plt.plot(Uni, Yx,  'r' )
plt.plot(X_Test, Y_Test, '.' 'k')
plt.show()

neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform',algorithm='auto')
neigh.fit(X_Train, Y_Train)
print("Training Accuracy",neigh.score(X_Train,Y_Train))
Yx = (neigh.predict(Uni))
plt.plot(Uni, Yx, 'r')
plt.plot(X_Train, Y_Train, '.' 'k' )
plt.title('3nn Classifier with Training data’ for k = 3')
plt.show()
print("Testing Accuracy",neigh.score(X_Test,Y_Test))
plt.title('3nn Classifier with Testing data’ for k = 3')
plt.plot(Uni, Yx,  'r' )
plt.plot(X_Test, Y_Test, '.' 'k')
plt.show()

neigh = KNeighborsClassifier(n_neighbors=9, weights='uniform',algorithm='auto')
neigh.fit(X_Train, Y_Train)
print("Training Accuracy",neigh.score(X_Train,Y_Train))

Yx = (neigh.predict(Uni))
plt.plot(Uni, Yx, 'r')
plt.plot(X_Train, Y_Train, '.' 'k' )
plt.title('9nn Classifier with Training data’ for k = 9')
plt.show()
print("Testing Accuracy",neigh.score(X_Test,Y_Test))
plt.title('9nn Classifier with Testing data’ for k = 9')
plt.plot(Uni, Yx,  'r' )
plt.plot(X_Test, Y_Test, '.' 'k')
plt.show()



###Training/Testing Accuracy(K)
K=[1,3,5,7,9,11,13,15]
TrainA = []
TestA = []
for i in range(1,16,2):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='uniform',algorithm='auto')
    neigh.fit(X_Train, Y_Train)
    TrainA.append(neigh.score(X_Train,Y_Train))
    TestA.append(neigh.score(X_Test,Y_Test))
plt.title('Training accuracy as a function of k')
plt.plot(K,TrainA)
plt.show()
plt.title('Testing accuracy as a function of k')
plt.plot(K,TestA)
plt.show()








