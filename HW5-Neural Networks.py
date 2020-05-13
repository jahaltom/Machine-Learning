import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

# load training set
ytrain = []
Xtrain = []
with open('HW3train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
        if len(row)==3:
            ytrain.append( int(row[0]) )
            Xtrain.append( [float(row[1]) , float(row[2]) ])
            
Xtrain = np.array(Xtrain)
ytrain = np.array(ytrain)

# load testing set
ytest = []
Xtest = []
with open('HW3test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',lineterminator='\n')
    for row in reader:
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





#####
learnrateinitrange = np.logspace(-3,-1,3)
alpharange = np.logspace(-6,0,5) 
Depth = [1,2,3,4,5,6,7,8,9,10]

BestTrain2 = []
BestTest2 = []
AlphaTrain2 = []
AlphaTest2 = []  
LearnTrain = []
LearnTest = []  
for i in range (1,11):
    BestTrain = []
    BestTest = []
    AlphaTrain = []
    AlphaTest = []
    for x in learnrateinitrange:  
        TestA = []
        TrainA = []
        for t in alpharange:
            clf = MLPClassifier(hidden_layer_sizes=(i),activation='relu', solver='sgd', learning_rate='adaptive', alpha = t, learning_rate_init=x, max_iter=200)
            clf.fit(Xtrain, ytrain)
            TestA.append(clf.score(Xtest,ytest))
            TrainA.append (clf.score(Xtrain,ytrain))
        BestTrain.append(max(TrainA))
        BestTest.append(max(TestA))
        AlphaTrain.append(TrainA.index(max(TrainA)))
        AlphaTest.append(TestA.index(max(TestA)))
    BestTrain2.append(max(BestTrain))
    LearnTrain.append(BestTrain.index(max(BestTrain)))
    indexTrain = BestTrain.index(max(BestTrain))
    AlphaTrain2.append(AlphaTrain[indexTrain])

    BestTest2.append(max(BestTest))
    LearnTest.append(BestTest.index(max(BestTest)))
    indexTest = BestTest.index(max(BestTest))
    AlphaTest2.append(AlphaTest[indexTest])
plt.title('Training accuracy as a function of number of nodes')
plt.plot(Depth,BestTrain2)
plt.show()
plt.title('Testing accuracy as a function of number of nodes')
plt.plot(Depth,BestTest2)
plt.show()    

print(BestTest2)
i = BestTest2.index(max(BestTest2)) + 1
t = alpharange[AlphaTest2[i-1]]
x = learnrateinitrange[LearnTest[i-1]]

clf = MLPClassifier(hidden_layer_sizes=(i),activation='relu', solver='sgd', learning_rate='adaptive', alpha = t, learning_rate_init=x, max_iter=200)
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
plt.title(i)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


# train network with a single hidden layer of 2 nodes

clf = MLPClassifier(hidden_layer_sizes=(2,),\
            activation='tanh', solver='sgd', \
            max_iter=1)

clf.fit(Xtrain, ytrain)
print(clf.coefs_)
print('\n\n')
print(clf.intercepts_)

w11 = np.linspace(1, -5, 55)
w21 = np.linspace(1, -5, 55)

# create a meshgrid and evaluate training MSE
W11, W21 = np.meshgrid(w11, w21)
MSEmesh = []
for coef1, coef2 in np.c_[W11.ravel(), W21.ravel()]:
    clf.coefs_[1][0][2] = coef1 
    clf.coefs_[0][0][1] = coef2
    MSEmesh.append( [clf.score(Xtrain,ytrain)] )

MSEmesh = np.array(MSEmesh)

# Put the result into a color plot
MSEmesh = MSEmesh.reshape(W11.shape)

ax = plt.axes(projection='3d')
ax.plot_surface(W11, W21, MSEmesh, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Training MSE');
plt.show()



###5
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


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

scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


learnrate = np.logspace(-2,-0.5,4)
alpharange = np.logspace(-6,0,4) 
noderange = range(5,100,5)

BestTrain2 = []
BestTest2 = []
AlphaTrain2 = []
AlphaTest2 = []  
LearnTrain = []
LearnTest = []  
for i in noderange:
    BestTrain = []
    BestTest = []
    AlphaTrain = []
    AlphaTest = []
    for x in learnrateinitrange:  
        TestA = []
        TrainA = []
        for t in alpharange:
            clf = MLPClassifier(hidden_layer_sizes=(i),activation='relu', solver='sgd', learning_rate='adaptive', alpha = t, learning_rate_init=x, max_iter=200)
            clf.fit(Xtrain, ytrain)
            TestA.append(clf.score(Xtest,ytest))
            TrainA.append (clf.score(Xtrain,ytrain))
        BestTrain.append(max(TrainA))
        BestTest.append(max(TestA))
        AlphaTrain.append(TrainA.index(max(TrainA)))
        AlphaTest.append(TestA.index(max(TestA)))
    BestTrain2.append(max(BestTrain))
    LearnTrain.append(BestTrain.index(max(BestTrain)))
    indexTrain = BestTrain.index(max(BestTrain))
    AlphaTrain2.append(AlphaTrain[indexTrain])

    BestTest2.append(max(BestTest))
    LearnTest.append(BestTest.index(max(BestTest)))
    indexTest = BestTest.index(max(BestTest))
    AlphaTest2.append(AlphaTest[indexTest])
plt.title('Training accuracy as a function of number of nodes')
plt.plot(noderange,BestTrain2)
plt.show()
plt.title('Testing accuracy as a function of number of nodes')
plt.plot(noderange,BestTest2)
plt.show()    
print(BestTrain2)
print(BestTest2)
i = BestTest2.index(max(BestTest2)) + 1
print(i)
x = learnrateinitrange[LearnTest[i-1]]
Depth = range(1,51)

TestA = []
TrainA = []
for s in range(1,51):
            clf = MLPClassifier(hidden_layer_sizes=(i),activation='relu', solver='sgd', learning_rate='adaptive', alpha = 0.0, learning_rate_init=x, max_iter=s)
            clf.partial_fit(Xtrain, ytrain,np.unique(ytrain))
            TestA.append(clf.score(Xtest,ytest))
            TrainA.append (clf.score(Xtrain,ytrain))

plt.title('Training accuracy as a function of number of iterations')
plt.plot(Depth,TrainA)
plt.show()
plt.title('Testing accuracy as a function of number of iterations')
plt.plot(Depth,TestA)
plt.show()   





