
library(RCurl)
library(e1071)
library(caret)
library(doSNOW)
library(ipred)
library(xgboost)
library(dplyr)
library(tidyr)
library(naniar)
library(corrplot)
library(psych)
library(grid)
library(ggplot2)
library(kknn)
library(pls)
library(pamr)
library(mda)
library(rattle)
library(vtreat)
library(glmnet)
library(zoo)
library(rsconnect)
library(gridExtra)
library(grid)
library(ggfortify)
library(magrittr)
library(lubridate)
library(ggpmisc)
library(DT)
library(doBy)


## ----trainfeatures_labels, class.output="scroll-300"--------------------------------------------
#Importing Datasets Into the R-Console

# Importing features dataset using "getURL" method from the RCurl package. 
# This dataset contains information about the various features that can affect the incidence of the cases of dengue per week.
trfeat <- getURL("https://s3.amazonaws.com/drivendata/data/44/public/dengue_features_train.csv")
trfeat <-read.csv(text = trfeat)
names(trfeat)
trfeat <- trfeat[, -c(4)]
trfeat$weekofyear <- as.factor(trfeat$weekofyear)
dim(trfeat)
DT::datatable(trfeat)


## -----------------------------------------------------------------------------------------------
# Weekly Total Number of Cases of Dengue in Each of the Two Cities by Year
# Importing the training data features and labels 
trlabel <- getURL("https://s3.amazonaws.com/drivendata/data/44/public/dengue_labels_train.csv")
trlabel <- read.csv(text = trlabel)
names(trlabel)
dim(trlabel)
trlabel$weekofyear <- as.factor(trlabel$weekofyear)
DT::datatable(trlabel)


## ----mergedtrainingset, class.output="scroll-300"-----------------------------------------------
# Merging features and labels by their composite keys (i.e., a combination of 'city', 'year' and 'week of year')
dengue_train <- merge(trfeat, trlabel, by=c("city", "year", "weekofyear"))
names(dengue_train)
dim(dengue_train)
DT::datatable(dengue_train)


## ----missing, class.output="scroll-300"---------------------------------------------------------
anyNA(dengue_train)
# Visualizing missing values for the training data
vis_miss(dengue_train)
gg_miss_var(dengue_train) + theme_minimal()
gg_miss_var(dengue_train, facet = city) + theme_gray()
ggplot(dengue_train, aes(x=ndvi_ne, y = total_cases)) + geom_point()
ggplot(dengue_train, aes(x=ndvi_ne, y = total_cases)) + geom_miss_point()


## ----impute, class.output="scroll-300"----------------------------------------------------------
# Imputing missing values by using 'last-observation carried forward' method
dengue_train <- na.locf(dengue_train)
anyNA(dengue_train)
vis_miss(dengue_train)
names(dengue_train)
describeBy(dengue_train[,c(4:24)], group = dengue_train$city)
dates <- make_datetime(year = dengue_train$year) + weeks(dengue_train$weekofyear)
dates <- as.Date(dates)
trcases<- ggplot(data=dengue_train, aes(x=dates, y= total_cases)) + geom_area(aes(color = city, fill=city),
                                                             alpha= 0.5, position = position_dodge(0.8)) +
                                              scale_color_manual(values = c("#00AF88", "#E7B800")) + 
                                              scale_fill_manual(values = c("#00AF88", "#E7B800"))
trcases + scale_x_date(date_labels = "%V/%Y") + facet_grid(. ~city)

#summaryBy(dengue_train[,-c(1:3)] ~ city, data = dengue_train, FUN=function(x)(c(mean=round(mean(x), digits = 1))))


# Randomization of the training data
random_index <- sample(1:nrow(dengue_train), nrow(dengue_train))
random_train <- dengue_train[random_index, ]
names(random_train)
dim(random_train)
anyNA(random_train)


## -----------------------------------------------------------------------------------------------
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)


## ----grid---------------------------------------------------------------------------------------
# Defining the tuning grid
grid <- expand.grid(eta = c(0.05, 0.5),
                         nrounds = c(70, 90),
                         max_depth = 1:6,
                         min_child_weight = c(1.0, 4),
                         colsample_bytree = c(0.5, 1),
                         gamma = c(10, 3, 0.1),
                         subsample = c(0.8, 1))


## ----traincontrol-------------------------------------------------------------------------------
# Defining trainControl for the ML Algorithms
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 5,
                              search = "grid")


## ----knn----------------------------------------------------------------------------------------
set.seed(45220)
model_kknn <- caret::train(total_cases ~ .,
                           data = random_train [,-c(2)],
                           type="prob",
                           method = "kknn",
                           tuneLength = 15,
                           preProcess = NULL,
                           trControl = train.control)
model_kknn


## ----glmnet-------------------------------------------------------------------------------------
# GLMNET Algorithm to Train The Prediction Model: generalized linear model via penalized maximum likelihood; the regulaization path is computed for elasticnet penalty at a grid of values for the regularization parameter lambada
set.seed(45220)
model_glmnet <- caret::train(total_cases ~ .,
                             data = random_train [,-c(2)],
                             method = "glmnet",
                             preProcess = NULL,
                             trControl = train.control)
model_glmnet


## ----rf-----------------------------------------------------------------------------------------
x <- random_train[,2:22]

metric <- "MAE"
mtry <- sqrt(ncol(x))
model_rf <- caret::train(total_cases ~ ., 
                         data = random_train [,-c(2)],
                         method = "rf",
                         preProcess = NULL,
                         metric = metric,
                         tuneGrid = expand.grid(.mtry = mtry),
                         trControl = train.control)
model_rf


## ----rpart--------------------------------------------------------------------------------------
set.seed(123)
model_rpart <- caret::train(total_cases ~ ., data = random_train [,-c(2)],
                               method = "rpart",
                               preProcess = NULL,
                               trControl = train.control)
model_rpart
summary(model_rpart)
fancyRpartPlot(model_rpart$finalModel)


## ----pls----------------------------------------------------------------------------------------
set.seed(27)
model_pls <- caret::train(total_cases ~ .,
                          data = random_train [,-c(2)],
                          method = "pls",
                          preProcess = NULL,
                          trControl = train.control)
model_pls


## ----xgb----------------------------------------------------------------------------------------
model_xgb <- caret::train(total_cases ~ .,
                          data = random_train [,-c(2)],
                          method = "xgbTree",
                          tuneGrid = grid,
                          trControl = train.control)
model_xgb


## ----final--------------------------------------------------------------------------------------
models <- list( 
                xgb = model_xgb,
                rf = model_rf, 
                glmnet = model_glmnet, 
                kknn = model_kknn, 
                pls = model_pls,
                tree = model_rpart
)
resample_results <- resamples(models)
summary(resample_results)


## ----testfeatures-------------------------------------------------------------------------------
# Importing the test data features on which the predictive model will be applied to predict total number of cases per week at a future date)
testset <- getURL("https://s3.amazonaws.com/drivendata/data/44/public/dengue_features_test.csv")
dengue_test <- read.csv(text=testset)
names(dengue_test)
dim(dengue_test)
dengue_test <- dengue_test[, -c(4)]
dim(dengue_test)
dengue_test$weekofyear <- as.factor(dengue_test$weekofyear)
DT::datatable(dengue_test)
# Visualizing missing values for the test data
anyNA(dengue_test)
vis_miss(dengue_test)


## ----impute_test--------------------------------------------------------------------------------
dengue_test <- na.locf(dengue_test)
anyNA(dengue_test)
vis_miss(dengue_test)


## ----predict------------------------------------------------------------------------------------
## Predicting total cases on test data
pred <- predict(model_xgb, dengue_test)
dengue_test$total_cases <- round(pred, digits = 0)



## ----visualize----------------------------------------------------------------------------------
# Visualizing the time-series total cases on the test data
dengue_test$dates <- as.Date(make_datetime(year = dengue_test$year) + weeks(dengue_test$weekofyear))
p<- ggplot(data=dengue_test, aes(x=dates, y= total_cases)) + geom_area(aes(color = city, fill=city),
                                                             alpha= 0.5, position = position_dodge(0.8)) +
                                              scale_color_manual(values = c("#00AF88", "#E7B800")) + 
                                              scale_fill_manual(values = c("#00AF88", "#E7B800"))
p + scale_x_date(date_labels = "%V/%Y") + facet_grid(. ~city)


## ----summary------------------------------------------------------------------------------------
# Summary of the predicted total cases
summary(dengue_test$total_cases)


## ----export-------------------------------------------------------------------------------------
#Entering the predicted 'total_cases' from the test-set into the submission form
Submitformat <- getURL("https://s3.amazonaws.com/drivendata/data/44/public/submission_format.csv")
submitformat <- read.csv(text=Submitformat)
submitformat$total_cases<- dengue_test$total_cases

# Exporting the output (total cases) to local drive as an Excel file
write.csv(submitformat, "D://STUDY//MSIS//DM//submit041520xgb_send.csv", row.names = FALSE)

