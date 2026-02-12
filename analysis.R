#We will practice on a marketing data set including information of the advertising budget on three
#advertising budget on three advertising media (youtube,facebook and newspaper) along with the 
#sales based on 200 advertising experiments

#We will apply two ML methods (kNN and linear regression) to predict the sales units using the 
#amounts of money spent on three medias, and explore the impacts of three advertising medias on 
#the sales.  

#We will follow this outline
#1. Familiarising with the dataset 
# -inspecting the summary of variables of interesting  
# -visualizing the relationship among variables by historgrams, scatter plots matrix and function corrplot()
#2. Split data into training and test sets 
#3. Perform both kNN and lm models 
#4. Check residuals to see if your linear regression model can be improved. If yes, improve it. 
#5. Comment on which model(s) you would choose by comparing their prediction performance in
#terms of RMSE and meaningful interpretation.

#We have to first load a few packages we will use
library(ggplot2)
install.packages("psych")
library(psych)
library(corrplot)
library(caret)

# Load the data
data(marketing, package = "datarium")

# 1. Fimilarising with the dataset
marketing
# The dataset consists of 4 variable names, namely: youtube, facebook, newspaper, sales
# and they corresponding to the following description:
# amount of money spent for advertisement on youtube, in thousand dollars 
# amount of money spent for advertisement on facebook, in thousand dollars 
# amount of money spent for advertisement on newspaper, in thousand dollars 
# sales units

dim(marketing)
summary(marketing)
ggplot(marketing, aes(sales)) + geom_histogram()

# We now visualise the relationship between any pair of the 4 variables in the dataset by
# matrix scatter plots, using R package "psyche", we create a matrix of scatter plots
# with bivarite scatter plots below the diagonal, the Pearson correlation above the diagonal
# and histograms on the diagonal
pairs.panels(marketing, 
             method = "pearson", 
             hist.col = "steelblue",
             pch = 21,
             density = TRUE,
             ellipses = FALSE)
# We observe that the variables youtube and facebook are highly related to sales,
# with the Pearson coorelation coefficient being 0.78 and 0.58 respectively
# while newspaper is not so much related to sales.

# Another way is to observe the correlation between the variables is to use the
# function corrplot in package "corrplot"
# we observe the same results as before
corrplot(cor(marketing),type="upper",method="color",addCoef.col = "black",number.cex = 0.6)

# 2. Split data into training and test sets 
set.seed(100)
training.idx <- sample(1:nrow(marketing), size = nrow(marketing)*0.8)
train.data <- marketing[training.idx, ]
test.data <- marketing[-training.idx, ]
dim(train.data)
# notice that the no. of rows in train.data=160, in step 3.1, we need to specify the number 
# folds in cross validation, an optimal size of each fold is 40-50, and since 160/4=40
# we will set that number to 4

# 3.1 Perform kNN method
set.seed(101)
knnmodel <- train(
  sales~., data = train.data, method = "knn",
  trControl = trainControl("cv", number = 4),
  preProcess = c("center", "scale"),
  tuneLength = 10
)

# Plot model error RMSE vs different values of k
plot(knnmodel)
# Best tuning parameter k 
knnmodel$bestTune
# k=5 is the one that minimizes the prediction error RMSE 


# Make predictions on the test data
knnpredictions <- predict(knnmodel, test.data)
# Compute the prediction error RMSE
RMSE(knnpredictions, test.data$sales)
#RMSE = 1.366957
# visualize the prediction results, we plot predicted sales vs sales in test data
plot(test.data$sales, knnpredictions,main="Prediction performance of kNN regression") 
# add a reference line x=y 
abline(0,1, col="red")
# Most data points are close to the 45 degree line, implying a very accurate prediction

# 3.2 Perform Linear Regression method
lmodel<- lm(sales~., data = train.data) 
summary(lmodel)
# R-squard=0.8925, 
# The linear regression model with 3 predictors explains 89% variation in the sales data

# Make predictions on the test data
lpredictions <- predict(lmodel, test.data)
# Compute the prediction error RMSE
RMSE(lpredictions, test.data$sales)
# RMSE = 1.95369, slighter bigger than that of kNN
# visualize the prediction results, we plot predicted sales vs sales in test data
plot(test.data$sales, lpredictions,main="Prediction performance of linear regression") 
abline(0,1, col="red")
# Linear regression makes an accurate prediction, but kNN still performs better

# Check residuals to see if your linear regression model can be improved. If yes, improve it.
# Look at the residuals plot
par(mfrow=c(2,2))
plot(lmodel)
# The residual plot shows an outlying point #131 and a quadratic pattern indicating that 2nd order predictors may be needed. 
# Therefore, we will improve the linear regression model by adding second order terms
# There are a few predictors in this example, we simply form 2nd order terms using those 
# predictors which are highly related to outcome only.
# Recall from the matrix of scatter plots and corrplot function,
# The predictors youtube and facebook are highly related to sales
# So we add 2nd order of those predictors in the linear regression
lmodel.2 <- lm(sales~youtube+facebook+newspaper+I(youtube^2)+I(facebook^2), data = train.data)
summary(lmodel.2)
# R-squared = 0.918
# Make predictions on test data
l2predictions <- predict(lmodel.2, test.data)
# Compute the prediction error RMSE
RMSE(l2predictions, test.data$sales)
# RMSE = 1.954943, which did not improve, this model does not improve the prediction

# We now consider adding another 2nd order term, a cross term capturing 
# the interaction between both predictors youtube and facebook
lmodel.3<- lm(sales~youtube+facebook+newspaper+I(youtube^2)+I(youtube*facebook)+I(facebook^2), data = train.data)
summary(lmodel.3)
# R-squared has increased to 0.984, this improved regression model can explain almost all variation in data.
l3predictions <-predict(lmodel.3, test.data)
RMSE(l3predictions, test.data$sales)
# RMSE hasdecreased to 0.5134645, a huge decrease from the kNN and previous regression models
par(mfrow=c(1,1))
plot(test.data$sales, l3predictions, main="Prediction performance of linear regression")
abline(0,1, col="red")
# All predicted values are very close to the actual values of sales in the test set
# Check residual plots
par(mfrow=c(2,2))
plot(lmodel.3)
# Residual plot now shows a horizontal line at 0 with no clear quadratic pattern,
# The fitting is good enough although there is still an outlier namely, 131

# 5. Comment on which model(s) you would choose by comparing their prediction performance in
#terms of RMSE and meaningful interpretation.
# We conclude the best model for prediction is the third linear regression model (lmodel.3)
# due to its very low RMSE

#######################################################################
# Additional comments: 
# We did not remove outlier 131 because it does not affect significantly the fitting of the models and predictions
# But here, we shall consider removing it

market <- marketing[-131, ]
set.seed(100)
training.idx <- sample(1:nrow(market), size = nrow(market)*0.8)
train.data <- market[training.idx, ]
test.data <- market[-training.idx, ]

set.seed(101)
knnmodel <- train(
  sales~., data = train.data, method = "knn",
  trControl = trainControl("cv", number = 3), #this time, we set each size of each fold to around 50
  preProcess = c("center", "scale"),
  tuneLength = 10
)
plot(knnmodel)
knnmodel$bestTune
# k=5
knnpredictions <- predict(knnmodel, test.data)
RMSE(knnpredictions, test.data$sales)
# RMSE = 1.537185
lmodel.4<- lm(sales~youtube+facebook+newspaper+I(youtube^2)+I(youtube*facebook)+I(facebook^2), data = train.data)
summary(lmodel.4)
# R-squard = 0.9901, a slight improvement from lmodel.3
l4predictions <- predict(lmodel.4, test.data)
RMSE(l4predictions, test.data$sales)
# RMSE = 0.5643404
par(mfrow=c(1,1))
plot(lmodel.4)
# The reesidul plot similarly shows a horizontal line at 0
# Conclusion: the outlier #131 does not affect significantly the model fitting and predictions
# We can either remove it or don't remove it.
