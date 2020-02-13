#read dataset 
svm <- read.csv(file.choose())
svm <- svm[0:310,3:12]
summary(svm)
#data preprocessing
svm$Sex<-as.integer(as.factor(svm$Sex))
svm$MaritalDesc<-as.integer(as.factor(svm$MaritalDesc))
svm$PerformanceScore<-as.integer(as.factor(svm$PerformanceScore))

#custom normalization function

#divide the data into train and test
svm_train<-svm[1:220,]
svm_test<-svm[221:310,]
# to train model
# e1071 package from LIBSVM library
# SVMlight algorithm klar package 
# kvsm() function uses gaussian RBF kernel 
# Building model 
library(kernlab)
library(caret)
model1<-ksvm(PerformanceScore ~.,data = svm_train,kernel = "vanilladot")
model1

help(kvsm)
# Different types of kernels 
# "rbfdot", "polydot", "tanhdot", "vanilladot", "laplacedot", 
# "besseldot", "anovadot", "splinedot", "matrix"

# kernel = rfdot 
model_rfdot<-ksvm(PerformanceScore ~.,data = svm_train,kernel = "rbfdot")
pred_rfdot<-predict(model_rfdot,newdata=svm_test)
mean(pred_rfdot==svm_test) 

# kernel = vanilladot
model_vanilla<-ksvm(PerformanceScore ~.,data = svm_train,kernel = "vanilladot")
pred_vanilla<-predict(model_vanilla,newdata=svm_test)
mean(pred_vanilla==svm_test)

# kernal = besseldot
model_besseldot<-ksvm(PerformanceScore ~.,data = svm_train,kernel = "besseldot")
pred_bessel<-predict(model_besseldot,newdata=svm_test)
mean(pred_bessel==svm_test)

# kernel = polydot

model_poly<-ksvm(PerformanceScore ~.,data = svm_train,kernel = "polydot")
pred_poly<-predict(model_poly,newdata = svm_test)
mean(pred_poly==svm_test) 


