This is the project of the Coursera course Practical Machine Learning.

Project Description
===================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt,
forearm, arm, and dumbell of 6 participants. They were asked to perform
barbell lifts correctly and incorrectly in 5 different ways.

The data consists of a Training data and a Test data (to be used to
validate the selected model).

The goal of your project is to predict the manner in which they did the
exercise. This is the “classe” variable in the training set. You may use
any of the other variables to predict with.

Note: The dataset used in this project is a courtesy of “Ugulino, W.;
Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable
Computing: Accelerometers’ Data Classification of Body Postures and
Movements”

Getting and Cleaning the Data
=============================

-   We first set the seed and load the data.

<!-- -->

    set.seed(123321)
    trainingRaw<-read.csv("pml-training.csv")
    testingRaw<-read.csv("pml-testing.csv")

-   Next, we do the data cleaning part. Firstly, we get rid of the first
    seven columns as they do not contribute to the prediction of the
    classe variable. Secondly, we eliminate the features that include
    NA’s as the machine learning algorithms are not suited to handle
    missing data.

<!-- -->

    trainingRaw<-trainingRaw[,-(1:7)]
    testingRaw<-testingRaw[,-(1:7)]
    trainingRaw<- trainingRaw[, colSums(is.na(trainingRaw)) == 0]
    testingRaw<- testingRaw[, colSums(is.na(testingRaw)) == 0]

-   Then, we divide the training data into two separate parts, one for
    training the algorithms and other one to test the algorithms’
    performance, called validation. Note that the initial testing data
    loaded is used in the final stage of the project.

<!-- -->

    inTrain <- createDataPartition(trainingRaw$classe, p = 0.7, list = F)
    training <- trainingRaw[inTrain, ]
    validation <- trainingRaw[-inTrain, ]

-   Finally, we eliminate the near zero variables, namely the variables
    which have almost no variation and would not contribute to the
    explaining of the classe variable.

<!-- -->

    nzvcols <- nearZeroVar(training)
    training <- training[,-nzvcols]
    validation <- validation[,-nzvcols]

Model Building and Prediction
=============================

-   We build two models, one is based on the random forest algorithm and
    the other is based on the gradient boosting algorithm, as they are
    both robust to correlation between variables and could handle a
    large number of predictor variables.

Random Forest Model
-------------------

We use 5-fold cross validation technique for the random forest
algorithm. After fitting the model, we predict using the validation data
frame we created from the raw training dataset downloaded for the
project and report the accuracy with other relevant information using
confusion matrix.

    control<-trainControl(method = "cv",number = 5,verboseIter = F)
    mod1<-train(classe~.,method="rf",data=training,trControl=control)
    pred1<-predict(mod1,validation)
    confusionMatrix(pred1,validation$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670    7    0    0    0
    ##          B    2 1125    3    0    1
    ##          C    1    7 1019    9    3
    ##          D    0    0    4  954    6
    ##          E    1    0    0    1 1072
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9924          
    ##                  95% CI : (0.9898, 0.9944)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9903          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9976   0.9877   0.9932   0.9896   0.9908
    ## Specificity            0.9983   0.9987   0.9959   0.9980   0.9996
    ## Pos Pred Value         0.9958   0.9947   0.9808   0.9896   0.9981
    ## Neg Pred Value         0.9990   0.9971   0.9986   0.9980   0.9979
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2838   0.1912   0.1732   0.1621   0.1822
    ## Detection Prevalence   0.2850   0.1922   0.1766   0.1638   0.1825
    ## Balanced Accuracy      0.9980   0.9932   0.9945   0.9938   0.9952

We see that we get a pretty solid accuracy of about 99.24% for this
model.

Gradient Boosting Model
-----------------------

We use repeated 5-fold cross validation technique for the gradient
boosting algorithm. After fitting the model, we predict using the
validation data frame we created from the raw training dataset
downloaded for the project and report the accuracy with other relevant
information using confusion matrix.

    control2 <- trainControl(method = "repeatedcv", number = 5, verboseIter = F,repeats=1)
    mod2<-train(classe~.,method="gbm",data=training,trControl=control2,verbose=F)
    pred2<-predict(mod2,validation)
    confusionMatrix(pred2,validation$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1637   39    0    0    2
    ##          B   23 1063   29    3   13
    ##          C   10   37  988   32   10
    ##          D    3    0    7  918   16
    ##          E    1    0    2   11 1041
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9596          
    ##                  95% CI : (0.9542, 0.9644)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9488          
    ##                                           
    ##  Mcnemar's Test P-Value : 1.506e-08       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9779   0.9333   0.9630   0.9523   0.9621
    ## Specificity            0.9903   0.9857   0.9817   0.9947   0.9971
    ## Pos Pred Value         0.9756   0.9399   0.9174   0.9725   0.9867
    ## Neg Pred Value         0.9912   0.9840   0.9921   0.9907   0.9915
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2782   0.1806   0.1679   0.1560   0.1769
    ## Detection Prevalence   0.2851   0.1922   0.1830   0.1604   0.1793
    ## Balanced Accuracy      0.9841   0.9595   0.9723   0.9735   0.9796

We observe that although we again get a very good accuracy of about
95.96%, the random forest model performs better even only by a slight
margin.

Prediction
----------

Thus, for the final testing of the original testing data of 20
observations, we predict using the random forest model, as the results
of the prediction is given below.

    finalpred<-predict(mod1,testingRaw)
    finalpred

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
