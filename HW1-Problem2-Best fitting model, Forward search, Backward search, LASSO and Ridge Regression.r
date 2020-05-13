###Make plots for all the data. MEDV=Median value of owner-occupied homes in $1000’s 

with(HouseData,plot(CRIM,MEDV))
title('CRIM')
with(HouseData,plot(ZN,MEDV))
title('ZN')
with(HouseData,plot(INDUS,MEDV))
title('INDUS')
with(HouseData,plot(CHAS,MEDV))
title('CHAS')
with(HouseData,plot(NOX,MEDV))
title('NOX')
with(HouseData,plot(RM,MEDV))
title('RM')
with(HouseData,plot(AGE,MEDV))
title('AGE')
with(HouseData,plot(DIS,MEDV))
title('DIS')
with(HouseData,plot(RAD,MEDV))
title('RAD')
with(HouseData,plot(TAX,MEDV))
title('TAX')
with(HouseData,plot(PTRATIO,MEDV))
title('PTRATIO')
with(HouseData,plot(B,MEDV))
title('B')
with(HouseData,plot(LSTAT,MEDV))
title('LSTAT')





##2b##
## Find the best fitting linear model for every subset of AGE, INDUS, NOX, RM, TAX using the first n = 400 samples. Reserve the remaining samples for validation. 
install.packages("leaps")
install.packages("car")
library(leaps)
library(car)
HouseData <- read.csv('housingdata.csv')
TrainingSet <- head(HouseData, 400)
leap <- regsubsets(MEDV~ AGE + INDUS + NOX + RM + TAX, data=TrainingSet, nbest=1, method = "exhaustive")

##2ci##
## For i = 0 to i = 5, select the model with i attributes from part (b). that has the lowest validation set MSE.
subsets(leap, statistic="rss")
##y=meanY  i=0 case##
m <- mean(TrainingSet[["MEDV"]])
MSE_Avg = mean((Y_Train-m)^2)
reg.summary= summary(leap) 
##2cii##Make a plot of the training data total loss of the best fitting model with i variables. 
R<- (reg.summary$rss)/400
MSE_T <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5), "MSE" = MSE_T)
plot(df, main="Training loss - best subsets", pch=19)
###Using the validation data set (samples not used in fitting), make a similar plot of the validation set’s MSE, titled “Validation loss - best subsets” .
ValdationSet <- tail(HouseData,106)
Model1 = lm(MEDV~ RM, data = ValdationSet)
Model2 = lm(MEDV~ RM + AGE, data = ValdationSet)
Model3 = lm(MEDV~ RM + AGE + TAX, data = ValdationSet)
Model4 = lm(MEDV~ RM + AGE + TAX + NOX, data = ValdationSet)
Model5 = lm(MEDV~ RM + AGE + TAX + NOX + INDUS, data = ValdationSet)
x<-anova(Model1,Model2,Model3,Model4,Model5)
R<- (x[, 2])/106
MSE_Avg = mean((Y_Test-m)^2)
MSE_V <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5), "MSE" = MSE_V)
plot(df, main="Validation loss - best subsets", pch=19)

##2Ciii###
## Now find the best fitting linear model for every subset of AGE, INDUS, NOX, RM, TAX using all the samples. Instead of using a validation set, we will use a complexity penalty. We will measure total complexity using Mallow’s Cp.
##Make a plot of the total complexity Cp as a function of the number of variables i, where for each i you are using the best fitting model that has i variables.
leap <- regsubsets(MEDV~ AGE + INDUS + NOX + RM + TAX, data=HouseData, nbest=1, method = "exhaustive")
subsets(leap, statistic="cp")

##2Civ########################################
install.packages("glmnet")
##Ridge###################

Y_Train <- data.matrix(TrainingSet[14], rownames.force = NA)
X_Train <- data.matrix(TrainingSet[,c(7,3,5,6,10)], rownames.force = NA)
Y_Test <- data.matrix(ValdationSet[14], rownames.force = NA)
X_Test <- data.matrix(ValdationSet[,c(7,3,5,6,10)], rownames.force = NA)
##grid=10^seq(10,-2, length =100) 

##Ridge <- glmnet(X_Train,Y_Train,alpha=0,lambda=grid, thresh=1e-12) 
Ridge <- glmnet(X_Train,Y_Train,alpha=0, thresh=1e-12) 
##MSE for all 100 lambda##
MSE <- vector(mode="numeric", length=0)
for(i in 1:100){
	ridge.pred <- predict(Ridge ,s=i,newx=X_Test) 
	MSE <- c(MSE, mean((ridge.pred - Y_Test)^2))
}
##L1 Norms and Lambdas##
L2 <- vector(mode="numeric", length=0)
for(i in 1:100){
	L2 <- c(L2,(sqrt(sum(coef(Ridge)[-1,i]^2))))
}
lambda <- Ridge$lambda

TC = MSE + lambda*L2

plot(lambda,TC)

##BestFit##
alpha.fit <- cv.glmnet(X_Train,Y_Train, type.measure="mse", alpha=0, family="gaussian")
alpha.predict <- predict(alpha.fit, s=alpha.fit$lambda.1se, newx=X_Test)
mean((alpha.predict - Y_Test)^2)

##LASSO####################

Y_Train <- data.matrix(TrainingSet[14], rownames.force = NA)
X_Train <- data.matrix(TrainingSet[,c(7,3,5,6,10)], rownames.force = NA)
Y_Test <- data.matrix(ValdationSet[14], rownames.force = NA)
X_Test <- data.matrix(ValdationSet[,c(7,3,5,6,10)], rownames.force = NA)
##grid=10^seq(10,-2, length =100) 


Lasso <- glmnet(X_Train,Y_Train,alpha=1) 





##MSE for all 100 lambda##
MSE <- vector(mode="numeric", length=0)
for(i in 1:69){
	lasso.pred <- predict(Lasso ,s=i,newx=X_Test) 
	MSE <- c(MSE, mean((lasso.pred - Y_Test)^2))
}
##L1 Norms and Lambdas##
L2 <- vector(mode="numeric", length=0)
for(i in 1:69){
	L2 <- c(L2,(sqrt(sum(coef(Lasso)[-1,i]^2))))
}
lambda <- Lasso$lambda

TC = MSE + lambda*L2

plot(lambda,TC)



##2D#############################################

##2Di##Forward Search


leap <- regsubsets(MEDV~., data=TrainingSet, nbest=1, method = "forward", nvmax=13)

##subsets(leap, statistic="rss")
##subsets(leap, statistic="rss",ylim=c(8500,9200))
####2Di.i###
##y=mean  i=0 case##
m <- mean(TrainingSet[["MEDV"]])
MSE_Avg = mean((Y_Train-m)^2)
reg.summary= summary(leap) 
R<- (reg.summary$rss)/400
MSE_TAll <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13), "MSE" = MSE_TAll)
plot(df, main="Training loss - best subsets", pch=19)
######2Di.ii
ValdationSet <- tail(HouseData,106)
Model1 = lm(MEDV~ RM, data = ValdationSet)
Model2 = lm(MEDV~ RM + LSTAT, data = ValdationSet)
Model3 = lm(MEDV~ RM + PTRATIO + LSTAT , data = ValdationSet)
Model4 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS, data = ValdationSet)
Model5 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX, data = ValdationSet)
Model6 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD, data = ValdationSet)
Model7 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM, data = ValdationSet)
Model8 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX, data = ValdationSet)
Model9 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX+ ZN, data = ValdationSet)
Model10 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS, data = ValdationSet)
Model11 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS, data = ValdationSet)
Model12 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS + B, data = ValdationSet)
Model13 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS + B +AGE, data = ValdationSet)
x<-anova(Model1,Model2,Model3,Model4,Model5,Model6,Model7,Model8,Model9,Model10,Model11,Model12,Model13)
R<- (x[, 2])/106
MSE_Avg = mean((Y_Test-m)^2)
MSE_VAll <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13), "MSE" = MSE_VAll)
plot(df, main="Validation loss - best subsets", pch=19)




##############2Dii##Backward Search

leap <- regsubsets(MEDV~., data=TrainingSet, nbest=1, method = "backward", nvmax=13)

##subsets(leap, statistic="rss")
##subsets(leap, statistic="rss",ylim=c(8500,9200))
####2Dii.i###
##y=mean  i=0 case##
m <- mean(TrainingSet[["MEDV"]])
MSE_Avg = mean((Y_Train-m)^2)
reg.summary= summary(leap) 
R<- (reg.summary$rss)/400
MSE_TAll <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13), "MSE" = MSE_TAll)
plot(df, main="Training loss - best subsets", pch=19)
######2Dii.ii
ValdationSet <- tail(HouseData,106)
Model1 = lm(MEDV~ RM, data = ValdationSet)
Model2 = lm(MEDV~ RM + LSTAT, data = ValdationSet)
Model3 = lm(MEDV~ RM + PTRATIO + LSTAT , data = ValdationSet)
Model4 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS, data = ValdationSet)
Model5 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX, data = ValdationSet)
Model6 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD, data = ValdationSet)
Model7 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM, data = ValdationSet)
Model8 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX, data = ValdationSet)
Model9 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX+ ZN, data = ValdationSet)
Model10 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS, data = ValdationSet)
Model11 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS, data = ValdationSet)
Model12 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS + B, data = ValdationSet)
Model13 = lm(MEDV~ RM + LSTAT + PTRATIO + DIS + NOX + RAD + CRIM + TAX + ZN + CHAS + INDUS + B +AGE, data = ValdationSet)
x<-anova(Model1,Model2,Model3,Model4,Model5,Model6,Model7,Model8,Model9,Model10,Model11,Model12,Model13)
R<- (x[, 2])/106
MSE_Avg = mean((Y_Test-m)^2)
MSE_VAll <- c(MSE_Avg,R)
df <- data.frame("Subset Size" = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13), "MSE" = MSE_VAll)
plot(df, main="Validation loss - best subsets", pch=19)

























#2Div#RIDGE!!!!!!!!!!###
Y_Train <- data.matrix(TrainingSet[14], rownames.force = NA)
X_Train <- data.matrix(TrainingSet[1:13], rownames.force = NA)
Y_Test <- data.matrix(ValdationSet[14], rownames.force = NA)
X_Test <- data.matrix(ValdationSet[1:13], rownames.force = NA)
##grid=10^seq(10,-2, length =100) 

##Ridge <- glmnet(X_Train,Y_Train,alpha=0,lambda=grid, thresh=1e-12) 
Ridge <- glmnet(X_Train,Y_Train,alpha=0, thresh=1e-12) 
##MSE for all 100 lambda##
MSE <- vector(mode="numeric", length=0)
for(i in 1:100){
	ridge.pred <- predict(Ridge ,s=i,newx=X_Test) 
	MSE <- c(MSE, mean((ridge.pred - Y_Test)^2))
}
##L1 Norms and Lambdas##
L2 <- vector(mode="numeric", length=0)
for(i in 1:100){
	L2 <- c(L2,(sqrt(sum(coef(Ridge)[-1,i]^2))))
}
lambda <- Ridge$lambda

TC = MSE + lambda*L2

plot(lambda,TC)

##BestFit##
alpha.fit <- cv.glmnet(X_Train,Y_Train, type.measure="mse", alpha=0, family="gaussian")
alpha.predict <- predict(alpha.fit, s=alpha.fit$lambda.1se, newx=X_Test)
mean((alpha.predict - Y_Test)^2)

##LASSO####################

Y_Train <- data.matrix(TrainingSet[14], rownames.force = NA)
X_Train <- data.matrix(TrainingSet[1:13], rownames.force = NA)
Y_Test <- data.matrix(ValdationSet[14], rownames.force = NA)
X_Test <- data.matrix(ValdationSet[1:13], rownames.force = NA)
##grid=10^seq(10,-2, length =100) 


Lasso <- glmnet(X_Train,Y_Train,alpha=1) 





##MSE for all 100 lambda##
MSE <- vector(mode="numeric", length=0)
for(i in 1:74){
	lasso.pred <- predict(Lasso ,s=i,newx=X_Test) 
	MSE <- c(MSE, mean((lasso.pred - Y_Test)^2))
}
##L1 Norms and Lambdas##
L2 <- vector(mode="numeric", length=0)
for(i in 1:74){
	L2 <- c(L2,(sqrt(sum(coef(Lasso)[-1,i]^2))))
}
lambda <- Lasso$lambda

TC = MSE + lambda*L2

plot(lambda,TC)