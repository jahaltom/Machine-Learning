# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:53:18 2020

@author: Jeff
"""

import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import pprint
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)



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


clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= 1)
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
plt.title('Depth = 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
    

clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= 2)
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
plt.title('Depth = 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
    


clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= 3)
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
plt.title('Depth = 3')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
    



clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= 4)
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
plt.title('Depth = 4')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
    
   
clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= 10)
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
plt.title('Depth = 10')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
    
 




TestA = []
TrainA = []
D = 0
Depth = []
for i in range(1, 11):
    clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= i)
    clf.fit(Xtrain, ytrain)
    TrainA.append (clf.score(Xtrain,ytrain))
    TestA.append(clf.score(Xtest,ytest)) 
    D = D + 1
    Depth.append(D)
    
plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,TrainA)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,TestA)
plt.show()
print(TestA)






##Bagging###

numtreerange=[1,5,10,25,50,100,200]

BestTrain = []
BestTest = []
TreeTrain = []
TreeTest = []
D = 0
Depth = []
for i in range (1,11):
    TestA = []
    TrainA = []
    for t in range(0,7):
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[t],max_features=1.0,criterion='gini',max_depth=i)
        clf.fit(Xtrain, ytrain)
        TestA.append(clf.score(Xtest,ytest))
        TrainA.append (clf.score(Xtrain,ytrain))
    BestTrain.append(max(TrainA))
    BestTest.append(max(TestA))
    TreeTrain.append(TrainA.index(max(TrainA)))
    TreeTest.append(TestA.index(max(TestA)))
    D = D + 1
    Depth.append(D)
    if i == 1:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[TreeTest[i-1]],max_features=1.0,criterion='gini',max_depth=1)
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
        plt.title('Depth = 1')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i == 2:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[TreeTest[i-1]],max_features=1.0,criterion='gini',max_depth=2)
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
        plt.title('Depth = 2')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i == 3:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[TreeTest[i-1]],max_features=1.0,criterion='gini',max_depth=3)
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
        plt.title('Depth = 3')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i == 4:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[TreeTest[i-1]],max_features=1.0,criterion='gini',max_depth=4)
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
        plt.title('Depth = 4')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i == 10:
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[TreeTest[i-1]],max_features=1.0,criterion='gini',max_depth=10)
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
        plt.title('Depth = 10')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,BestTrain)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,BestTest)
plt.show()    
print(BestTest)
print(TreeTest)






####Boosting###


            
##Find Best num tree
learnraterange=np.logspace(-3,0,15,base=10)
numtreerange=[1,5,10,25,50,100,200]


BestTrain2 = []
BestTest2 = []
TreeTrain2 = []
TreeTest2 = []  
LearnTrain = []
LearnTest = []  
for i in range (1,11):
    BestTrain = []
    BestTest = []
    TreeTrain = []
    TreeTest = []
    for x in learnraterange:  
        TestA = []
        TrainA = []
        for t in numtreerange:
            clf=GradientBoostingClassifier(learning_rate=x,n_estimators=t,max_depth=i)   
            clf.fit(Xtrain, ytrain)
            TestA.append(clf.score(Xtest,ytest))
            TrainA.append (clf.score(Xtrain,ytrain))
        BestTrain.append(max(TrainA))
        BestTest.append(max(TestA))
        TreeTrain.append(TrainA.index(max(TrainA)))
        TreeTest.append(TestA.index(max(TestA)))
    BestTrain2.append(max(BestTrain))
    LearnTrain.append(BestTrain.index(max(BestTrain)))
    indexTrain = BestTrain.index(max(BestTrain))
    TreeTrain2.append(TreeTrain[indexTrain])

    BestTest2.append(max(BestTest))
    LearnTest.append(BestTest.index(max(BestTest)))
    indexTest = BestTest.index(max(BestTest))
    TreeTest2.append(TreeTest[indexTest])
    if i ==1:
        clf=GradientBoostingClassifier(learning_rate=learnraterange[LearnTest[i-1]],n_estimators=numtreerange[TreeTest2[i-1]],max_depth=1)
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
        plt.title('Depth = 1')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i ==2:
        clf=GradientBoostingClassifier(learning_rate=learnraterange[LearnTest[i-1]],n_estimators=numtreerange[TreeTest2[i-1]],max_depth=2)
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
        plt.title('Depth = 2')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i ==3:
        clf=GradientBoostingClassifier(learning_rate=learnraterange[LearnTest[i-1]],n_estimators=numtreerange[TreeTest2[i-1]],max_depth=3)
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
        plt.title('Depth = 3')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i ==4:
        clf=GradientBoostingClassifier(learning_rate=learnraterange[LearnTest[i-1]],n_estimators=numtreerange[TreeTest2[i-1]],max_depth=4)
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
        plt.title('Depth = 4')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    if i ==10:
        clf=GradientBoostingClassifier(learning_rate=learnraterange[LearnTest[i-1]],n_estimators=numtreerange[TreeTest2[i-1]],max_depth=10)
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
        plt.title('Depth = 10')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    
plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,BestTrain2)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,BestTest2)
plt.show()    

print(BestTest2)
print(TreeTest2)  
    
    






####5###
ytrain = []
Xtrain = []
with open('digits-train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==48:
            ytrain.append( float(row[0]) )
            Xtrain.append((row[1:47]))    

                   
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)



  
    

ytest = []
Xtest = []
with open('digits-test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
#         print(row)
        if len(row)==48:
            ytest.append( float(row[0]) )
            Xtest.append((row[1:47]))    
            

Xtest = np.array(Xtest)
ytest = np.array(ytest)







#Report training and testing accuracies)
TestA = []
TrainA = []
D = 0
Depth = []
for i in range(1, 21):
    clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth= i)
    clf.fit(Xtrain, ytrain)
    TrainA.append (clf.score(Xtrain,ytrain))
    TestA.append(clf.score(Xtest,ytest)) 
    D = D + 1
    Depth.append(D)
    
plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,TrainA)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,TestA)
plt.show()
print(TestA)


##Bagging
numtreerange=[1,5,10,25,50,100,200]

BestTrain = []
BestTest = []
TreeTrain = []
TreeTest = []
D = 0
Depth = []
for i in range (1,21):
    TestA = []
    TrainA = []
    for t in range(0,7):
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[t],max_features=1.0,criterion='gini',max_depth=i)
        clf.fit(Xtrain, ytrain)
        TestA.append(clf.score(Xtest,ytest))
        TrainA.append (clf.score(Xtrain,ytrain))
    BestTrain.append(max(TrainA))
    BestTest.append(max(TestA))
    TreeTrain.append(TrainA.index(max(TrainA)))
    TreeTest.append(TestA.index(max(TestA)))
    D = D + 1
    Depth.append(D)

plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,BestTrain)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,BestTest)
plt.show()    
    
print(BestTest)
print(TreeTest)
    
    


##Find Best num tree and #Report training and testing accuracies)
numtreerange=[1,5,10,25,50,100,200]

BestTrain = []
BestTest = []
TreeTrain = []
TreeTest = []
D = 0
Depth = []
for i in range (1,21):
    TestA = []
    TrainA = []
    for t in range(0,7):
        clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[t],max_features='sqrt',criterion='gini',max_depth=i)
        clf.fit(Xtrain, ytrain)
        TestA.append(clf.score(Xtest,ytest))
        TrainA.append (clf.score(Xtrain,ytrain))
    BestTrain.append(max(TrainA))
    BestTest.append(max(TestA))
    TreeTrain.append(TrainA.index(max(TrainA)))
    TreeTest.append(TestA.index(max(TestA)))
    D = D + 1
    Depth.append(D)

plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,BestTrain)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,BestTest)
plt.show()    
    
print(BestTest)
print(TreeTest)


##Boosting
learnraterange=np.logspace(-2,0,10,base=10)
numtreerange=[50,100,150]
Depth = [1,2,3,4,5,6]

BestTrain2 = []
BestTest2 = []
TreeTrain2 = []
TreeTest2 = [] 
LearnTrain = []
LearnTest = []  
for i in range (1,7):
    BestTrain = []
    BestTest = []
    TreeTrain = []
    TreeTest = []
    for x in learnraterange:  
        TestA = []
        TrainA = []
        for t in numtreerange:
            clf=GradientBoostingClassifier(learning_rate=x,n_estimators=t,max_depth=i)   
            clf.fit(Xtrain, ytrain)
            TestA.append(clf.score(Xtest,ytest))
            TrainA.append (clf.score(Xtrain,ytrain))
        BestTrain.append(max(TrainA))
        BestTest.append(max(TestA))
        TreeTrain.append(TrainA.index(max(TrainA)))
        TreeTest.append(TestA.index(max(TestA)))
    BestTrain2.append(max(BestTrain))
    LearnTrain.append(BestTrain.index(max(BestTrain)))
    indexTrain = BestTrain.index(max(BestTrain))
    TreeTrain2.append(TreeTrain[indexTrain])
    
    BestTest2.append(max(BestTest))
    LearnTest.append(BestTest.index(max(BestTest)))
    indexTest = BestTest.index(max(BestTest))
    TreeTest2.append(TreeTest[indexTest])

plt.title('Training accuracy as a function of Depth')
plt.plot(Depth,BestTrain2)
plt.show()
plt.title('Testing accuracy as a function of Depth')
plt.plot(Depth,BestTest2)
plt.show()    

print(BestTest2)
print(TreeTest2)



  
    




