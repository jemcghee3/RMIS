## load the excel file as a dataframe
library(readxl)
library(tidyverse)


data.raw <- read_excel("fall_2022_manual.xlsm")
data.clean <- data.raw
summary(data.clean)

# drop rows where column ...13 is NA
data.clean <- data.clean[!is.na(data.clean$...13),]
#convert column 13 to float
data.clean$...13 <- as.numeric(data.clean$...13)
data.clean <- data.clean[!is.na(data.clean$...13),]

# drop rows where column ...18 or ...19 is NA
data.clean <- data.clean[!is.na(data.clean$...18),]
data.clean <- data.clean[!is.na(data.clean$...19),]

# set column ...18 to 1 if both column ...18 and ...19 are "accept"; that is, both accept
data.clean$...18 <- ifelse(data.clean$...18 == "accept" & data.clean$...19 == "accept", 1, 0)
# turn column ...18 into a factor
data.clean$...18 <- as.factor(data.clean$...18)

# split data.clean into test and training sets
set.seed(123)
train.index <- sample(1:nrow(data.clean), size = 0.8*nrow(data.clean))
train <- data.clean[train.index,]
test <- data.clean[-train.index,]



# calculate the hyperparameters of a logistic regression model using 5-fold cross-validation
library(caret)
ctrl <- trainControl(method = "cv", number = 5)
glmFit <- train(...18 ~ ...13, data = train, method = "glmnet", trControl = ctrl, family = "binomial")
glmFit


# perform logistic regression with column ...18 as the y variable and column ...13 as the x
glm.fit <- glm(...18 ~ ...13, data = train, family = binomial)
summary(glm.fit)

# predict the probability of acceptance for the test set
prob <- predict(glm.fit, newdata = test, type = "response")
# convert the probability to a binary value
pred <- ifelse(prob > 0.5, 1, 0)

# calculate the accuracy of the model
mean(pred == test$...18)

# calculate the confusion matrix
table(pred, test$...18)

