# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:53:18 2020

@author: Jeff
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

x1 = np.linspace(-10,10,100)
x2 = -x1
plt.plot(x1, x2, '-r')
plt.plot([0,0],[-10,10],'b')
plt.plot([-10,10],[0,0],'g')
plt.xlabel('X1', color='#1C2833')
plt.ylabel('X2', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()


x1 = np.linspace(-10,10,100)
x2 = 1+3*x1
plt.plot(x1, x2, '-r')

plt.legend(loc='upper left')
plt.grid()


x1 = np.linspace(-10,10,100)
x2 = 1-x1/2
plt.plot(x1, x2, '-r')
plt.xlabel('X1', color='#1C2833')
plt.ylabel('X2', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()


ytrain = []
Xtrain = []
with open('HW3train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==3:
            ytrain.append( int(row[0]) )
            Xtrain.append( [float(row[1]) , float(row[2]) ])

Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)


for x,y in zip(Xtrain, ytrain):

    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Train')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

ytest = []
Xtest = []
with open('HW3test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==3:
            ytest.append( int(row[0]) )
            Xtest.append( [float(row[1]) , float(row[2]) ])

Xtest = np.array(Xtest)
ytest = np.array(ytest)


for x,y in zip(Xtest, ytest):

    if y==1:
        col = 'blue'
    if y==2:
        col = 'red'
    if y==3:
        col = 'black'
    plt.scatter(x[0], x[1],  color=col)
    
plt.title('Scatterplot HW3Test')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()



h = .03  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh 
x1_min, x1_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
x2_min, x2_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
x1mesh, x2mesh = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))



# Create color maps
cmap_light = ListedColormap(['lightblue', 'lightcoral', 'grey'])
cmap_bold = ListedColormap(['blue', 'red', 'black'])

for n_neighbors in [22]:
    # we create an instance of Neighbours Classifier and fit the data.
    
    clf = KNeighborsClassifier(n_neighbors, weights='uniform',algorithm='auto')
    clf.fit(Xtrain, ytrain)

    Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytrain_colors = [y-1 for y in ytrain]
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Training Set' % (n_neighbors))
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    # Plot the testing points with the mesh
    ypred = clf.predict(Xtest)
    plt.figure()
    plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
    ytest_colors = [y-1 for y in ytest]
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('%i-NN Testing Set' % (n_neighbors))
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    #Report training and testing accuracies
    print('Working on k=%i'%(n_neighbors))
    trainacc =  clf.score(Xtrain,ytrain) 
    testacc = clf.score(Xtest,ytest) 
    print('\tThe training accuracy is %.2f'%(trainacc))
    print('\tThe testing accuracy is %.2f'%(testacc))
    
    
    
    
    
    ###Training/Testing Accuracy(K)
K=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
TrainA = []
TestA = []
for i in range(1,31):
    neigh = KNeighborsClassifier(n_neighbors=i, weights='uniform',algorithm='auto')
    neigh.fit(Xtrain, ytrain)
    TrainA.append(neigh.score(Xtrain,ytrain))
    TestA.append(neigh.score(Xtest,ytest))
plt.title('Training accuracy as a function of k')
plt.plot(K,TrainA)
plt.show()
plt.title('Testing accuracy as a function of k')
plt.plot(K,TestA)
plt.show()
plt.title('Training accuracy as a function of k')
plt.plot(K,TrainA)
plt.show()

##LDA
clf=LinearDiscriminantAnalysis(solver='svd',shrinkage=None,priors= None)
clf.fit(Xtrain, ytrain)
Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytrain_colors = [y-1 for y in ytrain]
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Training Set')
plt.xlabel('X1')
plt.ylabel('X2')
    
    # Plot the testing points with the mesh
ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y-1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Testing Set')
plt.xlabel('X1')
plt.ylabel('X2')
    
    #Report training and testing accuracies
print('Working on k=%i'%(n_neighbors))
trainacc =  clf.score(Xtrain,ytrain) 
testacc = clf.score(Xtest,ytest) 
print('\tThe training accuracy is %.2f'%(trainacc))
print('\tThe testing accuracy is %.2f'%(testacc))

##QDA
clf=QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0)
clf.fit(Xtrain, ytrain)
Z = clf.predict(np.c_[x1mesh.ravel(), x2mesh.ravel()])

    # Put the result into a color plot
Z = Z.reshape(x1mesh.shape)

    # Plot the training points with the mesh
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytrain_colors = [y-1 for y in ytrain]
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain_colors, cmap=cmap_bold, s=20)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Training Set')
plt.xlabel('X1')
plt.ylabel('X2')
    
    # Plot the testing points with the mesh
ypred = clf.predict(Xtest)
plt.figure()
plt.pcolormesh(x1mesh, x2mesh, Z, cmap=cmap_light)
ytest_colors = [y-1 for y in ytest]
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest_colors, cmap=cmap_bold, s=30)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Testing Set')
plt.xlabel('X1')
plt.ylabel('X2')
    
    #Report training and testing accuracies
print('Working on k=%i'%(n_neighbors))
trainacc =  clf.score(Xtrain,ytrain) 
testacc = clf.score(Xtest,ytest) 
print('\tThe training accuracy is %.2f'%(trainacc))
print('\tThe testing accuracy is %.2f'%(testacc))


##SVM
#Cvals=np.logspace(-4,2,25,base=10)

#for i in Cvals:
#    clf=SVC(C=i,kernel='poly',degree=1,gamma=1.0,coef0=1.0, shrinking=True,probability=False,max_iter=1000)