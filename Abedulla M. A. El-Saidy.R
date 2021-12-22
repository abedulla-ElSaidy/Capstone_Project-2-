#install.packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(earth)) install.packages("earth", repos = "http://cran.us.r-project.org")
if(!require(pls)) install.packages("pls", repos = "http://cran.us.r-project.org")
if(!require(earth)) install.packages("earth", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")

#load .packages
library(tidyverse)
library(caret)
library(randomForest)
library(pls)
library(earth)
library(glmnet)
library(mlbench)


# Download Audi cars data from https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=hyundi.csv
# Load the data on R as CSV file
#audi_cars =read.csv("~/R/z/pr2/pr2/Data/audi.csv",sep = ",")

#split the dataset into training and validation

# 1:  row numbers 
#RowNum = createDataPartition(audi_cars$price, p=0.80, list=FALSE)

# 2: Create the training  dataset
#train_data <- audi_cars[RowNum,]

#  3: Create the test dataset
#test_data <- audi_cars[-RowNum,]

# a copy of the data
#write.csv(train_data,"~/R/z/pr2/pr2/Data/train_data.csv")
#write.csv(test_data,"~/R/z/pr2/pr2/Data/test_data.csv")
#rm(RowNum)

#Load Data
train_data =read.csv("~/R/z/pr2/Data/train_data.csv",sep = ",")
test_data =read.csv("~/R/z/pr2/Data/test_data.csv",sep = ",")

# Data charcterstics
summary(audi_cars)
glimpse(audi_cars)
dim(audi_cars)


#check na values
sum(is.na(audi_cars))
#no na values


#Visualizations to understand the data

#Price per year
audi_cars %>% mutate(Year = as.numeric(year)) %>% 
  ggplot() +   geom_smooth(aes(x=Year, y=price),method="loess") + 
  ggtitle("Price per year")
audi_cars %>% mutate(Year = as.factor(year)) %>% 
  ggplot() +   geom_boxplot(aes(x=Year, y=price)) + 
  ggtitle("Price per year")
#We can conclude the more the year, the more the price

#Price per model
audi_cars %>% mutate(Model = (model)) %>% 
  ggplot() +   geom_boxplot(aes(x=reorder(Model,price), y=((price)))) + 
  labs(x="Model", y="price")+
  ggtitle("Price per model")
#We can conclude that some models have higher price like R8

#Price per transmission
pie( summary(as.factor(audi_cars$transmission)),col=rainbow(5))
audi_cars %>% mutate(TR = (transmission)) %>% 
  ggplot() +   geom_boxplot(aes(x=reorder(TR,price), y=((price)))) + 
  labs(x="transmission", y="price")+
  ggtitle("Price per transmission")

#We can conclude that Manual cheaper than automatic

#Price per milage
audi_cars %>% ggplot() +   geom_smooth(aes(x=mileage, y=price)) + 
  ggtitle("Price per milage")
#We can conclude the more milage the little price

#Price per fuelType
audi_cars %>% mutate(fuel = (fuelType)) %>% 
  ggplot() +   geom_boxplot(aes(x=reorder(fuel,price), y=((price)))) + 
  labs(x="fuelType", y="price")+
  ggtitle("Price per fuelType")
#We can conclude that Hybrid is more expensive

#Price per tax
audi_cars %>% ggplot() +   geom_smooth(aes(x=tax, y=price)) + 
  ggtitle("Price per tax")
#We can conclude no obvious relation between price and tax

#Price per mpg
audi_cars %>% ggplot() +   geom_smooth(aes(x=mpg, y=price)) + 
  ggtitle("Price per mpg")
#We can conclude inverse relation between price and mpg

#Price per enginSize
audi_cars %>% mutate(enginsize = (engineSize)) %>% 
  ggplot() +   geom_boxplot(aes(x=reorder(enginsize,price), y=((price)))) + 
  labs(x="enginsize", y="price")+
  ggtitle("Price per enginsize")
#We can conclude that high engin size costs alot

#  parameters correlation
audi_cars %>% select(where(is.numeric)) %>% cor() 
# the most effective parameters to price are = year+mileage+mpg+engineSize


# Model 1 Base Line mean only
m1_rm= RMSE(mean(train_data$price),train_data$price)
m1_rm

#model2_lm => price ~ year + mileage + mpg + engine Size
cv <- trainControl(  method = "repeatedcv",  number = 10,  repeats = 5)
model_lm = train(price ~ year+mileage+mpg+engineSize , data=train_data, method='lm',trControl = cv)
fit_lm = predict(model_lm,test_data)
summary(model_lm)
varimp = varImp(model_lm)
plot(varimp, main="variable importance")
lm2_RM = RMSE(test_data$price, fit_lm )
lm2_RM
comparison_lm=head(data.frame(Actual=test_data$price,Predicted=fit_lm))
comparison_lm

#model3_lm => price ~  engineSize
model_lm_engin = train(price ~ engineSize , data=train_data, method='lm')
fit_lm_engin = predict(model_lm_engin,test_data)
summary(model_lm_engin)
lm3_RM = RMSE(test_data$price, fit_lm_engin )
lm3_RM
comparison_lm_engin=head(data.frame(Actual=test_data$price,Predicted=fit_lm_engin))
comparison_lm_engin

#model4_lm => price ~  year*mileage*mpg*engineSize 
model_lm_h = train(price ~ year*mileage*mpg*engineSize , data=train_data, method='lm')
fit_lm_h = predict(model_lm_h,test_data)
summary(model_lm_engin)
lm4_RM = RMSE(test_data$price, fit_lm_h )
lm4_RM
comparison_lm_h=head(data.frame(Actual=test_data$price,Predicted=fit_lm_h))
comparison_lm_h

#model5_polynomial => price ~ polynomial
model_poly = train(price ~ (model)+poly(year,3)+poly(mileage,3)+poly(mpg,3)+poly(engineSize,3) , data=train_data, method='lm',trControl = cv)
fit_poly = predict(model_poly,test_data)
lm5_RM = RMSE(test_data$price, fit_poly )
lm5_RM
comparison_poly=head(data.frame(Actual=test_data$price,Predicted=fit_poly))
comparison_poly

#model6_gam => price ~ year+mileage+mpg+engineSize 
model_gam = train(price ~ (year+mileage+mpg+engineSize) , data=train_data, method='gam',trControl = cv)
fit_gam = predict(model_gam,test_data)
summary(model_gam)
gam6_RM = RMSE(test_data$price, fit_gam )
gam6_RM
comparison_gam=head(data.frame(Actual=test_data$price,Predicted=fit_gam))
comparison_gam


#model7_pls => price ~  all 
model_pls = train(price ~ . , data=train_data, method='pls',trControl = cv,tuneLength = 30)
fit_pls = predict(model_pls,test_data)
pls7_RM = RMSE(test_data$price, fit_pls )
pls7_RM
comparison_pls=head(data.frame(Actual=test_data$price,Predicted=fit_pls))
comparison_pls

#model8.randomForest
rf <- randomForest(price~., data=train_data, importance=TRUE,ntree=100,trControl = cv) 
fit_rf <- predict(rf, test_data)
varImpPlot(rf)
plot(rf)
rf8_RM = RMSE(test_data$price, fit_rf )
rf8_RM


#model9 Decision Tree 
TRE = train(price~., data=train_data, method="rpart",trControl = cv)
(TRE)
fit_TRE = predict(TRE, test_data)
TRE9_RM = RMSE(test_data$price, fit_TRE)
TRE9_RM


#model 10.k-Nearest Neighbors
knn = train(price~., data=train_data, method="knn",trControl = cv)
knn
plot(knn)
fit_knn <- predict(knn, test_data)
knn10_rm = RMSE(test_data$price, fit_knn )
knn10_rm


#model 11.Earth
model_earth = train(price ~ ., data=train_data, method='earth',trControl = cv)
fit_earth = predict(model_earth, test_data)
summary(model_earth)
varimp<- varImp(model_earth)
plot(varimp, main="Variable Importance")
plot(model_earth, main="Model Accuracies")
earth11_rm = RMSE(test_data$price, fit_earth )
earth11_rm




#model12_glm => price ~ all
model_glm = train(price ~ . , data=train_data, method='glm',trControl = cv)
fit_glm = predict(model_glm,test_data)
glm12_RM = RMSE(test_data$price, fit_glm )
glm12_RM
comparison_glm=head(data.frame(Actual=test_data$price,Predicted=fit_glm))
comparison_glm

#model13_lasso => price ~ all
model_lasso = train(price ~ . , data=train_data, method = 'glmnet', tuneGrid = expand.grid(alpha = 1, lambda = 1),trControl = cv)
fit_lasso = predict(model_lasso,test_data)
lasso13_RM = RMSE(test_data$price, fit_lasso )
lasso13_RM
comparison_lasso=head(data.frame(Actual=test_data$price,Predicted=fit_lasso))
comparison_lasso


#Results
Rmse_Result=data.frame(method=c("Model 1 - Base Line", 
                                "Model 2 (lm)", 
                                "Model 3 (lm)", 
                                "Model 4 (lm)", 
                                "Model 5 (lm)", 
                                "Model 6 (Generalized Additive- gam)", 
                                "Model 7 (Partial Least Squares-pls)", 
                                "Model 8 (randomForest)", 
                                "Model 9 (Decision Tree)" ,
                                "Model 10 (k-Nearest Neighbors)", 
                                "Model 11 (Multivariate Adaptive- Earth)", 
                                "Model 12 (Generalized Linear Model- glm)", 
                                "Model 13 (The lasso)"),
                       RMSE=c(m1_rm,
                              lm2_RM,
                              lm3_RM,
                              lm4_RM,
                              lm5_RM,
                              gam6_RM,
                              pls7_RM,
                              rf8_RM,
                              TRE9_RM, 
                              knn10_rm, 
                              earth11_rm,
                              glm12_RM,
                              lasso13_RM))
arrange(Rmse_Result,(RMSE))

