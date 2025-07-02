# ==============================================================================
# BC2407 Analytics 2
# Purpose:      Predicting Dyslipidemia With Machine Learning Models and 
#               Identifying Key Factors
# Disease:      Dyslipidemia
# Authors:      Amirul Haziq Bin Muhammad Effendy (U2210956K), 
#               Andrhea Angelina Therese Gaerlan San Gabriel (U2210291B),
#               Charan Kumar Velu (U2110392E),
#               Muhammad Wisnu Darmawan (U2220618B),
#               Yeung WenBin (U2110374B)
# DOC:          13-03-2024
# Topics:       Data Wrangling, Logistic Regression, CART, Random Forest
# Data Source:  pone.0243103.s001.sav
# Link:         https://plos.figshare.com/articles/dataset/Used_dataset_for_dyslipidemia_2020_/13814789
# Packages:     tidyverse, ggplot2, haven, labelled, caTools, caret, ROCR, pROC,
#               rpart, rpart.plot, randomForest

# ==============================================================================
# ============================= TABLE OF CONTENTS ==============================

# Part 1: Data Preparation & Cleaning

# Part 2: Data Exploration & Selecting Variables
# ------- 2.1 Data Visualisation
# ------- 2.2 Kruskal Test (For Continuous Variables)
# ------- 2.3 Chi-Squared test (For Categorical variables)
# ------- 2.4 Selecting Variables

# Part 3: Model Evaluation
# ------- 3.1 Train-Test Split & Set Threshold
# ------- 3.2 Model 1: Logistic Regression w Backward Elimination + Evaluation
# ------- 3.3 Model 2: CART + Evaluation
# ------- 3.4 Model 3: Random Forest + Evaluation

# ==============================================================================

# Set your working directory here :)
setwd('/Users/darmawan/Downloads/NTU Year 2 Sem 2/BC2407/Group Project')


# Automation to install packages that might have yet to be installed
package.list <- c("tidyverse", "ggplot2", "haven", "labelled", "caTools", 
                  "caret", "ROCR", "pROC", "rpart", "rpart.plot", "randomForest")
new.packages <- package.list[!(package.list %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(tidyverse)
library(ggplot2)
library(haven)
library(labelled)
library(caTools)
library(caret)
library(ROCR)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)





# ================== Part 1: Data Preparation & Cleaning =======================
# Load dataset, convert from sav to data frame
lipid.df <- to_factor(read_sav('pone.0243103.s001.sav'))
lipid.df <- data.frame(lipid.df)
View(lipid.df)
summary(lipid.df)


# Check for NA values
colSums(is.na(lipid.df))


# Target variable is: Dyslipidemia_outcome_2





# ============== Part 2: Data Exploration & Selecting Variables ================
# Creation of visualisations will be done to analyse the impact that certain
# variables have on dyslipidemia outcome.

# Not all variables will be used in the dataset to train and test models.
# The team will leverage various methods to evaluate the significance of each
# variable.

# Method 1: Using domain knowledge
#           e.g. low-density lipoprotein cholesterol [LDL_C] is a known
#           indicator of dyslipidemia. It should be added into dataset.

# Method 2: Kruskal-Wallis test
#           Determines if there is statistically significant difference in
#           medians of >= 3 independent groups. Can be applied to applied to
#           categorical Y and continuous X.

# Method 3: Chi-squared test
#           Measures the association between categorical Y and X
#           If p-value > threshold, there is strong association between
#           variables.





# ==============================================================================
# --------------------------- 2.1 Data Visualisation ---------------------------
# Set main colours to be used across plots
colour1 = '#F8766D' 
colour2 = '#00BFC4'


# Can physical activity alone help to reduce chances of dyslipidemia?
lipid2.df <- mutate(lipid2.df, 
                    total_activity_perweek = vigrous_intensity_activity_perweek 
                    + moderate_intensity_activity_perweek)

ggplot(data = lipid2.df, aes(fill=Dyslipidemia_outcome_2, y=..count.., 
                             x=total_activity_perweek)) + 
  geom_bar(position="stack", stat="count") +   
  stat_count(aes(y=after_stat(count), label=after_stat(count)), size = 3, 
             geom="text", position = position_stack(vjust=0.5)) + 
  labs(x = "Total Activity Per Week (Hrs)", y = "Number Of Surveyees", 
       title = "Physical Activity Alone Is Not Enough To Combat Dyslipidemia") +
  scale_fill_manual(values = c(colour2, colour1))


lipid2.df <- subset(lipid2.df, select = -total_activity_perweek)


# Does smoking have an effect on LDL-C levels and cause dyslipidemia?
lipid2.df <- mutate(lipid2.df, 
                    consume_alcohol = ifelse(alcohol_number_perweek == 0, 
                                             'Does Not Drink Alcohol', 
                                             'Drinks Alcohol'))

ggplot(data = lipid2.df, aes(smoke_years, LDL_C, colour = Dyslipidemia_outcome_2)) +
  geom_point() +
  labs(x = "Years Smoking", 
       y = "Low-density Lipoprotein Cholesterol (LDL-C) (mg/dL)",
       title = "Effects Of Smoking And Alcohol Consumption On Dyslipidemia",
       subtitle = 'Regardless of alcohol consumption status, smoking increases risk of hyperlipidemia') +
  scale_colour_manual(values = c(colour2, colour1)) +
  facet_grid(cols = vars(consume_alcohol))

lipid2.df <- subset(lipid2.df, select = -consume_alcohol)


# Does age and gender affect the chances of having dyslipidemia?
ggplot(data = lipid2.df, aes(fill=Dyslipidemia_outcome_2, y=..count.., x=Age)) + 
  geom_bar(position="stack", stat="count") +   
  stat_count(aes(y=after_stat(count), label=after_stat(count)), size = 3, geom="text", 
             position = position_stack(vjust=0.5)) + 
  labs(x = "Age", 
       y = "Number Of Surveyees",
       title = "Distribution Of Dyslipidemia By Age And Sex",
       subtitle = 'Women in their 50s - 70s are more at risk of dyslipidemia') +
  scale_fill_manual(values = c(colour2, colour1)) +
  facet_grid(rows = vars(Sex))





# ------------------------------ 2.2 Kruskal Test ------------------------------
# For categorical Y and continuous X
# Create dataframe to store the p-values of all continuous X variables
results.kruskal.df <- data.frame(variable = character(), 
                                 p_value = numeric(), 
                                 stringsAsFactors = FALSE)


# Create dataframe with only continuous X variables, and categorical Y
numeric.columns <- sapply(lipid.df, is.numeric)
numeric.data <- lipid.df[, numeric.columns]
numeric.data["Dyslipidemia_outcome_2"] <- lipid.df$Dyslipidemia_outcome_2


# Run kruskal test for variables
for (col in colnames(numeric.data)) {

  # Extract the column data
  column.data <- numeric.data[[col]]
  
  # Perform Kruskal-Wallis test
  kruskal_result <- kruskal.test(Dyslipidemia_outcome_2 ~ column.data, 
                                 data = numeric.data)
  
  # Store the results
  results.kruskal.df <- rbind(results.kruskal.df, 
                              data.frame(variable = col, 
                                         p_value = round(kruskal_result$p.value, 3)
                                         ))
}


# Remove Dyslipidemia_outcome_2 variable, View p-values of continuous X variables
results.kruskal.df <- results.kruskal.df[results.kruskal.df$variable != "Dyslipidemia_outcome_2",]
View(results.kruskal.df)


# Return only variables where p-value < 0.05
results.kruskal.df[results.kruskal.df$p_value < 0.05,]





# ---------------------------- 2.3 Chi-Squared Test ----------------------------
# For categorical Y and categorical X
# Create data frame to store the p-values of all categorical X variables
results.chisq.df <- data.frame(variable = character(), 
                               p_value = numeric(), 
                               stringsAsFactors = FALSE)


# Create dataframe with only categorical X and Y variables
factor.columns <- sapply(lipid.df, is.factor)
factor.data <- lipid.df[, factor.columns]


# Run chi-sq test for variables
for (col in colnames(factor.data)) {
  
  # Extract the column data
  chisq.table <- table(factor.data[col])

  # Perform chisq test
  chisq.result <- chisq.test(chisq.table)
  
  # Store the results
  results.chisq.df <- rbind(results.chisq.df, 
                            data.frame(variable = col, 
                                       p_value = round(chisq.result$p.value, 3)
                                       ))
}


# View p-values of continuous X variables
View(results.chisq.df)


# Return only variables where p-value < 0.05
results.chisq.df[results.chisq.df$p_value < 0.05,]





# -------------------------- 2.4 Selecting Variables ---------------------------
# Legend for explanation:
# DK - variable chosen because of domain knowledge
# KT - variable chosen because p-value < threshold in kruskal test
# CT - variable chosen because p-value < threshold in chi-sq test
lipid2.df <- lipid.df %>% select("Dyslipidemia_outcome_2", # Target variable
                                 "Total_Cholesterol",                   # DK
                                 "Fating_Blood_Glucose",                # DK
                                 "Triglyceride",                        # DK
                                 "HDL_C",                               # DK
                                 "HDL_Cutoff",                          # DK, KT
                                 "LDL_C",                               # DK
                                 "Sex",                                 # DK
                                 "Age",                                 # DK, KT
                                 "Height_cat",                          # CT
                                 "Weght",                               # DK, KT
                                 "SBP",                                 # DK
                                 "DBP",                                 # DK, KT
                                 "Blood_pressure_4cat",                 # DK, CT
                                 "Waist_circumference",                 # KT
                                 "W_to_Hip_ratio",                      # KT
                                 "BMI",                                 # DK
                                 "occupation",                          # CT
                                 "smoke_current",                       # DK, CT
                                 "smoke_years",                         # DK
                                 "live_with_smoker",                    # DK, CT
                                 "alcohol_number_perweek",              # DK
                                 "fruits_N_perweek",                    # DK
                                 "vegetable_N_perweek",                 # DK
                                 "vigrous_intensity_activity_perweek",  # DK
                                 "moderate_intensity_activity_perweek", # DK
                                 "vigrous_intensity_sports_perweek",    # DK
                                 "moderate_intensity_sports_perweek",   # DK
                                 "sitting_hours_perday",                # DK
                                 ) %>% replace(is.na(.), 0)
View(lipid2.df)
summary(lipid2.df)    ## 29 variables have been selected, inc. target variable
sum(is.na(lipid2.df)) ## No missing values





# ==================== Machine Learning Starts From Here :) ====================
# ==================== Part 3: Implement & Evaluate Models =====================
 
# This is the overview for Part 3:
# 3.1 - Train-Test Split & Set Threshold
# 3.2 - Model 1: Logistic Regression w Backward Elimination + Evaluation
# 3.3 - Model 2: Classification And Regression Tree (CART) + Evaluation
# 3.4 - Model 3: Random Forest + Evaluation

# All models will be evaluated using the 5 criteria:
# 1. Accuracy
# 2. Area under curve (AUC)
# 3. Precision ( true positives / [true positives + false positives] )
# 4. Recall    ( true positives / [true positives + false negatives] )
# 5. F1 Score  ( [2 * precision * recall] / [precision + recall] )

# NOTE: 
# Results that are displayed (e.g. model accuracy / important variables) were
# from testing session conducted on 26-03-24, 1500HRS.





# ==============================================================================
# ----------------- Part 3.1: Train-Test Split & Set Threshold -----------------
set.seed(2024)


# 70-30 train-test split will be applied across models
traintest_split <- sample.split(Y = lipid2.df$Dyslipidemia_outcome_2, 
                                SplitRatio = 0.7)
trainset <- subset(x = lipid2.df, subset = traintest_split == TRUE)
testset <- subset(x = lipid2.df, subset = traintest_split == FALSE)


# Set the threshold
threshold <- 0.5


# Create empty data frame to store test set results for models
testset.model.compare <- data.frame('Model' = c("Logistic Regression with BE", 
                                                "CART", "Random Forest"),
                                    'Accuracy' = rep(0, 3), 'AUC' = rep(0, 3),
                                    'Precision' = rep(0, 3), 'Recall' = rep(0, 3), 
                                    'F1 Score' = rep(0, 3))





# ----- Part 3.2: Logistic Regression w Backward Elimination + Evaluation ------
# Training model on trainset
log.model <- step(glm(Dyslipidemia_outcome_2 ~ ., 
                      family = binomial, 
                      data = trainset), direction = 'backward')


# Important variables
summary(log.model)

## Final model used 4 variables:
## Total_Cholesterol / Triglyceride / HDL_Cutoff / LDL_C


# Accuracy of model on train set
trainset.prob <- predict(log.model, type = 'response')
trainset.predict <- ifelse(trainset.prob > threshold, "yes", "no") 
## Comparing predicted probability of dyslipidemia_outcome with threshold

cm.trainset <- table(Actual = trainset$Dyslipidemia_outcome_2 , 
                     Predicted = trainset.predict, deparse.level = 2)
cm.trainset
mean(trainset.predict == trainset$Dyslipidemia_outcome_2) ## Overall accuracy = 1.0


# Evaluating model on test set - accuracy 
testset.prob <- predict(log.model, type = 'response', newdata = testset)
testset.predict <- ifelse(testset.prob > threshold, "yes", "no")
## Comparing predicted probability of dyslipidemia_outcome with threshold

cm.testset <- table(Actual = testset$Dyslipidemia_outcome_2 , 
                    Predicted = testset.predict, deparse.level = 2)
cm.testset
mean(testset.predict == testset$Dyslipidemia_outcome_2) ## Overall accuracy = 0.9479167
testset.model.compare[1,2] <- mean(testset.predict == testset$Dyslipidemia_outcome_2)


# For the prediction of developing dyslipidemia (positive - have dyslipidemia, negative - no dyslipidemia)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on test set - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[1,4] <- precision
testset.model.compare[1,5] <- recall
testset.model.compare[1,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC
pred = prediction(testset.prob, testset$Dyslipidemia_outcome_2)
auc = performance(pred, measure = "auc")
testset.model.compare[1,3] <- auc@y.values


# Evaluating model on test set - plot AUC
testset.predict.roc <- ifelse(testset.prob > threshold, 1, 0)
roc_score = roc(testset$Dyslipidemia_outcome_2, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")


# Q-Q Plot
par(mfrow= c(2,2))
plot(log.model)
par(mfrow= c(1,1))

## top left = model w outlier
## top right = plots assumption 2, that error has normal distribution (w/o outlier)
## bottom left = assumption 3, if variance is constant (w outlier)
## bottom right = influential outlier (regression w/o outlier)

## From top right graph, assumption 2 does not apply to logistic regression model, as
## The residuals do not follow normal distribution





# ------------------------ Part 3.3: CART + Evaluation -------------------------
# Training model on train set
cart.model <- rpart(Dyslipidemia_outcome_2 ~ . , data = trainset, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0)) 

## to grow max tree, minsplit = 2 and cp = 0


# Plot maximal tree
rpart.plot(cart.model, nn=T, main = "Maximal Tree For Trainset")


# Display pruning sequence and 10-fold CV errors as a chart
plotcp(cart.model, main ="Subtrees in dyslipidemia") 


# Display pruning sequence and 10-fold CV errors as a table
printcp(cart.model)


# Source: Prof. Neumann Chew's Textbook, Chapter 8 - CART
# Create function to find the optimal CP
optimal.cp <- function(cartmodel) {
  
  # Compute min CVerror + 1SE in maximal tree cart.model
  CVerror.cap <- cartmodel$cptable[which.min(cartmodel$cptable[,"xerror"]), "xerror"] + 
    cartmodel$cptable[which.min(cartmodel$cptable[,"xerror"]), "xstd"]
  
  # Find the optimal CP region where CV error is just below CVerror.cap in maximal tree cart model
  i <- 1; j<- 4
  while (cartmodel$cptable[i,j] > CVerror.cap) {
    i <- i + 1
  }
  
  # Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
  cp.optimal = ifelse(i > 1, sqrt(cartmodel$cptable[i,1] * cartmodel$cptable[i-1,1]), 1) # calculate geometric mean here
  
  return(cp.optimal)
}


# Find the optimal CP
cp.opt <- optimal.cp(cart.model)


# Prune the max tree
prune.cart.model <- prune(cart.model, cp = cp.opt)
print(prune.cart.model)
rpart.plot(prune.cart.model, nn=T, main = "Optimal Tree for Trainset")

## 3 variables used: LDL_C / Triglyceride / HDL_Cutoff


# Accuracy of model on train set
cart.yhat <- predict(prune.cart.model, type= 'class')
cm.trainset <- table(Actual = trainset$Dyslipidemia_outcome_2 , 
                     Predicted = cart.yhat, deparse.level = 2)
cm.trainset
mean(cart.yhat == trainset$Dyslipidemia_outcome_2) ## Overall Accuracy - 0.9955556


# Evaluating model on test set - accuracy 
prune.cart.model.yhat <- predict(prune.cart.model, type= 'class', newdata = testset)
cm.testset <- table(Actual = testset$Dyslipidemia_outcome_2 , 
                    Predicted = prune.cart.model.yhat, deparse.level = 2)
cm.testset
mean(prune.cart.model.yhat == testset$Dyslipidemia_outcome_2) ## Overall accuracy = 1
testset.model.compare[2,2] <- mean(prune.cart.model.yhat == testset$Dyslipidemia_outcome_2)


# For the prediction of developing dyslipidemia (positive - have dyslipidemia, negative - no dyslipidemia)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[2,4] <- precision
testset.model.compare[2,5] <- recall
testset.model.compare[2,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(prune.cart.model, type= 'prob', newdata = testset))
pred = prediction(testset.prob$yes, testset$Dyslipidemia_outcome_2)
auc = performance(pred, measure = "auc")
testset.model.compare[2,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$yes > threshold, 1, 0)
roc_score = roc(testset$Dyslipidemia_outcome_2, testset.predict.roc)
plot(roc_score, main = "AUC for CART")





# ------------------- Part 3.4: Random Forest + Evaluation ---------------------
# Training model on train set
rf.model <- randomForest(Dyslipidemia_outcome_2 ~ . , data = trainset, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model on train set
rf.model

## OOB estimate of  error rate: 4.89%
## Confusion matrix:
##     no yes class.error
## no  65  10  0.1066667
## yes  1 149  0.0000000

print((65+149) / (65+10+1+149)) ## Overall accuracy = 0.9511111


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Dyslipidemia Dataset
     Up Till 500 Trees")
## OOB error rate stabilised before 500 trees


# Evaluating model on test set - accuracy 
rf.model.yhat <- predict(rf.model, newdata = testset)
cm.testset <- table(Actual = testset$Dyslipidemia_outcome_2 , 
                    Predicted = rf.model.yhat, deparse.level = 2)
cm.testset
mean(rf.model.yhat == testset$Dyslipidemia_outcome_2) ## Overall accuracy = 0.9375
testset.model.compare[3,2] <- mean(rf.model.yhat == testset$Dyslipidemia_outcome_2)


# For the prediction of developing dyslipidemia (positive - have dyslipidemia, negative - no dyslipidemia)
# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[2,2]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[1,1]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[3,4] <- precision
testset.model.compare[3,5] <- recall
testset.model.compare[3,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(rf.model, type = 'prob', newdata = testset))
pred = prediction(testset.prob$yes, testset$Dyslipidemia_outcome_2)
auc = performance(pred, measure = "auc")
testset.model.compare[3,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$yes > threshold, 1, 0)
roc_score = roc(testset$Dyslipidemia_outcome_2, testset.predict.roc)
plot(roc_score, main = "AUC for Random Forest")


# Evaluation of models
testset.model.compare

##                         Model  Accuracy       AUC Precision   Recall  F1.Score
## 1 Logistic Regression with BE 0.9479167 0.9453125 0.9682540 0.953125 0.9606299
## 2                        CART 1.0000000 1.0000000 1.0000000 1.000000 1.0000000
## 3               Random Forest 0.9375000 0.9902344 0.9264706 0.984375 0.9545455

## In healthcare, a false negative is more detrimental than a false positive.
## As recall performance of Random Forest and CART are slightly better than
## Logistic Regression w BE, either Random Forest or CART is preferred.

## Between CART and Random Forest, CART is more appropriate for THIS DATASET.

## However, in real life, there would be more features present in dataset. 
## As dataset becomes more complex with more variables, 
## a Random Forest would be preferred to the CART model.






# -------------- Part 3.4.2: Random Forest (Using Entire Dataset) --------------
# Apply RF on entire lipid2.df dataset
rf.model <- randomForest(Dyslipidemia_outcome_2 ~ . , data = lipid2.df, 
                         na.action = na.omit, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model
rf.model

## OOB estimate of  error rate: 1.87%
## Confusion matrix:
##      no yes class.error
## no  101   6  0.05607477
## yes   0 214  0.00000000

## Compared to previous RF error rate (4.89%), current RF error rate is much
## less (1.87%)

print((101+214) / (101+6+0+214)) ## Overall accuracy = 0.9813084
print(101 / (101 + 0)) ## Recall = 1

testset.model.compare[3,2] ## Overall accuracy = 0.9375
testset.model.compare[3,5] ## Recall = 0.984375

## Both overall accuracy and recall is higher in new RF model compared to original
## RF model. Thus the new RF model would be preferred over the original model.


# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Dyslipidemia Dataset
     Up Till 500 Trees")
## OOB error rate stabilised before 500 trees


# Evaluating Other Properties Of Random Forest
rf.model$oob.times # Check how many times each sample has been OOB

inbag <- rf.model$inbag # Check each case, whether it is inbag or OOB in which tree
View(inbag)

rf.model$votes


# Important variables
var.impt <- importance(rf.model)
varImpPlot(rf.model, type = 1)

## Comparing variable importance of different models:

## Logistic Regression w BE used 4 variables:
## Total_Cholesterol / Triglyceride / HDL_Cutoff / LDL_C

## CART used 3 variables:
## LDL_C / Triglyceride / HDL_Cutoff

## Top 4 variables w most importance in Random Forest:
## Triglyceride / LDL_C / Total_Cholesterol / HDL_Cutoff

## The 3 models applied all identified common variables as important factors 
## these factors have to do with the fat content in a patient's blood.
## lifestyle factors like ones diet and physical activity are known to influence
## the fat content in a patient's blood.





# ===================== Thank You For Running This Script! =====================
# ======================= Have A Wonderful Day Ahead :) ========================
# ==================================== End =====================================