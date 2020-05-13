import numpy as np
import matplotlib.pyplot as plt
import statistics


#(a,b)#Ground Truth: If we have p + 1 data points, we can fit that perfectly with a pth order polynomial. Fit a third order polynomial to the following four points. 
x = np.array([0, 5, 8, 12])
y = np.array([10, 5, 12, 0])
z = np.polyfit(x, y, 3)
p = np.poly1d(z)
xp = np.linspace(-5, 20, 100)
#plot points and polynomial
plt.ylim(-5, 20)
plt.xlim(-5, 20)
plt.plot(x, y, '.' 'k', xp, p(xp), 'r' )
plt.plot( xp, p(xp), 'r', label="Ground-Truth"  )
plt.legend(loc="upper left")
plt.show()

#(d)# Use a uniform distribution over the interval [0,15] to gaenerate 30 random x values. Use Ploynomial to generate y values.
#Add in Noise to y values that is independent and identically distributed. Use the normal distribution, N(0,10). Repeat this for the 1000 and 5000 uniformaly distrubited random x values to be used later. 
GT0 = (p.c)[3]
GT1 = (p.c)[2]
GT2 = (p.c)[1]
GT3 = (p.c)[0]

s = np.random.uniform(0,15,30)
s2 = np.random.uniform(0,15,1000)
s3 = np.random.uniform(0,15,5000)
F = p(s)
F2 = p(s2)
F3 = p(s3)
mu, sigma = 0, 10 
N = np.random.normal(mu, sigma, 30)
N2 = np.random.normal(mu, sigma, 1000)
N3 = np.random.normal(mu, sigma, 5000)
Y_act = F+N
Y_act2 = F2 + N2
Y_act3 = F3 + N3
##Fit polynomial to the 30 data points and plot. 
plt.plot(s, Y_act, '.' 'k', xp, p(xp), 'r' )
plt.show()
##Mean square error for ground truth model
GTMSE = []
for j in range(30):
    GTMSE.append(abs(Y_act[j]-F[j])**2)
GTMSE2 = (np.sum(GTMSE)/30)

#(e)#
##Mean Squared Error##
y_pred = np.random.uniform(-10,20,100)
length = y_pred.size
length2 = Y_act.size
MSE2 = []
for i in range(length):
    MSE = []
    for j in range(length2):
       MSE.append(abs(Y_act[j]-y_pred[i])**2)
    MSE2.append(np.sum(MSE)/30)
plt.plot(y_pred, MSE2, '.' 'k', label="Mean Squared Error")
plt.legend(loc="upper left")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('MSE', fontsize=16)
plt.show()


##Mean Absolute Error##
length = y_pred.size
length2 = Y_act.size
MAE2 = []
for i in range(length):
    MAE = []
    for j in range(length2):
       MAE.append(abs(Y_act[j]-y_pred[i]))
    MAE2.append(np.sum(MAE)/30)
plt.plot(y_pred, MAE2, '.' 'k', label="Mean Absolute Error")
plt.legend(loc="upper left")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('MAE', fontsize=16)
plt.show()


##Special Total-Loss Function##
STL = []
length = y_pred.size
length2 = Y_act.size
for i in range(length):
    array = []
    for j in range(length2):
        res = Y_act[j] - y_pred[i]
        if res <0:
            res = (-1/5)*res
        else:
            res = 10*res
        array.append(res/((abs(s[j]-5))+0.01))
    STL.append(np.sum(array) )
plt.plot(y_pred, STL, '.' 'k', label="Special Total-Loss Function")
plt.legend(loc="upper left")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.show()

##g##
print("Mean:", np.sum(Y_act)/30)
print("Median:" , statistics.median(Y_act))

##h## Fit the data with polynomials from order 0 to order 30. 
##To evaluate over-fitting, use validation. Use the first 20 samples for fitting, reserving the rest as a validation set. 

#h2## Plot the MSE for the 20 samples used to fit for each polynomial for p = 0 to p = 30. In this plot, include a green horizontal line for the ground-truth model’s MSE on the training data. 
TrainX = s[:20] 
TrainY = Y_act[:20]
polyOrd = []
poly = []
MSE2 = []
for x in range(31):
    polyOrd.append(x)
    z = np.polyfit(TrainX, TrainY, x)
    p = np.poly1d(z)
    MSE = []
    y_pred = p(TrainX)
    poly.append(p)
    for i in range (20):
        MSE.append(abs(TrainY[i]-y_pred[i])**2)
    MSE2.append(np.sum(MSE)/20)

plt.plot(polyOrd, MSE2, '.' 'k', label="Training Loss")
plt.axhline(y=GTMSE2, color='g', linestyle='-')
plt.legend(loc="upper right")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('MSE', fontsize=16)
plt.show()


#h4##Plot the MSE for the 10 samples held out. In this plot, include a green horizontal line for the ground-truth model’s MSE on the validation data. 
TestX = s[20:30] 
TestY= Y_act[20:30]
MSE2 = []
for x in range(31):
    MSE = []
    y_pred = poly[x](TestX) 
    for i in range (10):
        MSE.append(abs(TestY[i]-y_pred[i])**2)
    MSE2.append(np.sum(MSE)/10)
    

plt.plot(polyOrd, MSE2, '.' 'k', label="Validation Testing Loss")
plt.axhline(y=GTMSE2, color='g', linestyle='-')
plt.legend(loc="upper right")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('MSE', fontsize=16)
plt.show()

##h6##Generate a new data-set with n = 1000(Made previously). Make another plot, titled “GroundTruth Testing Loss n = 1000” with the MSE of the polynomials fitted to the n = 30 data set evaluated on this new, larger data-set.
#Include a green horizontal line for the ground-truth model’s MSE.

TrainX = s2 
TrainY = Y_act2
MSE2 = []
for x in range(31):
    MSE = []
    y_pred = poly[x](TrainX) 
    for i in range (1000):
        MSE.append(abs(TrainY[i]-y_pred[i])**2)
    MSE2.append(np.sum(MSE)/1000)
    

plt.plot(polyOrd, MSE2, '.' 'k', label="Ground-Truth Testing Loss n=1000")
plt.axhline(y=GTMSE2, color='g', linestyle='-')
plt.legend(loc="upper right")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('MSE', fontsize=16)
plt.show()


##h8##Model Complexity. 
##Fit polynomials using all n = 30 data points and keep track of the corresponding MSE.
##The Calculate: Total Complexity = Total Training Loss + Lambda · Model Complexity. Model Complexity is polynomial order, Total Training Loss is MSE and Lambda is a threshold we choose.
## Pick a Lambda such that the model with p = 30 has the same total-complexity as the model with p = 0. Lambda= (MSE(0)-MSE(30))/30 
TrainX = s[:30] 
TrainY = Y_act[:30]
polyOrd = []
poly = []
MSE2 = []
for x in range(31):
    polyOrd.append(x)
    z = np.polyfit(TrainX, TrainY, x)
    p = np.poly1d(z)
    MSE = []
    y_pred = p(TrainX)
    poly.append(p)
    for i in range(30):
        MSE.append(abs(TrainY[i]-y_pred[i])**2)
    MSE2.append(np.sum(MSE)/30)
Lambda = (MSE2[0]-MSE2[30])/30
TC = []
for j in range(31):
    TC.append(MSE2[j] + polyOrd[j]*Lambda)
##Make a plot of the total-complexity as a function of the model order p. Use the title “Total Complexity p” 
plt.plot(polyOrd, TC, '.' 'k', label="Total Complexity p")
plt.legend(loc="upper center")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('Total Complexity', fontsize=16)
plt.show()


##h9## 
## Reusing the fitting from part (h).viii, select a new lambda and make new plots for penalties other than (Lambda · Model Complexity)
##Pick a Lambda such that the model with p = 30 has the same total-complexity as the model with p = 0.
##L1##Lambda1 * Summation(|ai|^1)      a=coefficient
##L2##Lambda2 * Summation(|ai|^2)      a=coefficient
SUM0 = (abs(poly[0].c))
SUM30 = np.sum(abs(poly[30].c))
Lambda = (MSE2[0] - MSE2[30])/(SUM0 + SUM30)
TC = []
for j in range(31):
    TC.append(MSE2[j] + Lambda*np.sum(abs(poly[j].c)))
plt.plot(polyOrd, TC, '.' 'k', label="Total Complexity L1")
plt.legend(loc="upper center")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('Total Complexity', fontsize=16)
plt.show()

SUM0 = (abs(poly[0].c)**2)
SUM30 = np.sum(abs(poly[30].c)**2)
Lambda = (MSE2[0] - MSE2[30])/(SUM0 + SUM30)
TC = []
for j in range(31):
    TC.append(MSE2[j] + Lambda*np.sum(abs(poly[j].c)**2))
plt.plot(polyOrd, TC, '.' 'k', label="Total Complexity L2")
plt.legend(loc="upper center")
plt.xlabel('Polynomial Order', fontsize=18)
plt.ylabel('Total Complexity', fontsize=16)
plt.show()



##i##
##Generate n = 5000 samples which will be used for fitting. Done previously. 
##For i = 1,...,100, fit the best p = 30 polynomial using n = 50i samples from the 5000 sample data set.
##Plot each coefficient {a0,a1..a30} as a function of 50i in separate plots. (so one plot for a0, etc). In each plot, draw a green line for the corresponding coefficient in the ground-truth polynomial. 
X_Ax = []
for c in range (50,5050, 50):
    X_Ax.append(c)
    
TrainX = s3 
TrainY = Y_act3
poly= []
a0 = []
for x in range(0, 5000, 50):
    XX = TrainX[:x+50]
    YY = TrainY[:x+50]
    z = np.polyfit(XX, YY, 30)
    p = np.poly1d(z)
    poly.append(p)

for T in range(31):
    for i in range (100):
        a0.append((poly[i].c)[T])
    plt.plot(X_Ax, a0, '.' 'k', label= 30 - T)
    plt.legend(loc="upper left")
    plt.xlabel('50i', fontsize=18)
    plt.ylabel('coefficient', fontsize=16) 
    plt.axhline(y=GT0, color='g', linestyle='-') 
    plt.show()
    a0 = []
 


