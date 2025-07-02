# ==============================================================================
# BC2407 Analytics 2
# Purpose:      Predicting Dyslipidemia With Machine Learning Models and 
#               Identifying Key Factors
# Disease:      Hypertension
# Authors:      Amirul Haziq Bin Muhammad Effendy (U2210956K), 
#               Andrhea Angelina Therese Gaerlan San Gabriel (U2210291B),
#               Charan Kumar Velu (U2110392E),
#               Muhammad Wisnu Darmawan (U2220618B),
#               Yeung WenBin (U2110374B)
# DOC:          13-03-2024
# Topics:       Data Wrangling, Logistic Regression, CART, Random Forest
# Data Source:  hypertension_data
# Link:         https://www.kaggle.com/datasets/prosperchuks/health-dataset
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

# Part 1: Data Cleaning

# Part 2: Data Exploration
# ------- 2.1 Data Visualisation (Basic Visualisation, can be polished further)
# ------- 2.2 Chi-Squared test (For Categorical variables)
# ------- 2.3 Kruskal Test (For Continuous Variables)
# ------- 2.4 Selected Variables

# Part 3: Model Evaluation
# --------3.1 - train-test split
# --------3.2 - Model 1: Logistic Regression w Backward Elimination + Evaluation
# --------3.3 - Model 2: Random Forest + Evaluation

##Set your working directory here :)
setwd('/Users/ywb/Desktop/Univeristy/AY 23:24/BC2407-Analytics 2/Project Work/Submission')

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





#======================== Part 1 : Data Cleaning ===============================
set.seed(2024)

data1 <- read.csv("hypertension_data.csv", header=TRUE)

summary(data1)

data1 <- na.omit(data1)

summary(data1)

str(data1)





#===================== Part 2 : Data Exploration ===============================
# ---------------------- 2.1 Data Visualisation --------------------------------

#Data Visualisation (Each Variable against target)
library(ggplot2)

#Age boxplot (No visible differenes: likely insignificant)

age1 <- ggplot(data1, aes(x = factor(target), y = age)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "Age", title = "Box Plot of Age by Target")

age1

#Age histogram of those with hypertension 
age2 <- ggplot(data = data1, aes(x = age)) +
  geom_histogram(data = subset(data1, target == 1),
                 binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "Age", y = "Frequency", title = "Age Distribution for Target = 1")

age2

#Sex boxplot
sex1 <- ggplot(data1, aes(x = factor(sex), fill = factor(target))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon")) +  # Custom colors for target
  labs(x = "Sex", y = "Count", fill = "Target", title = "Distribution of Target Across Sex") +
  theme_minimal()

sex1

#ChestPains Visualisation
cp1 <- ggplot(data1, aes(x = factor(target), fill = factor(cp))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon", "2"="yellow" , "3"= "grey")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "Chest Pain Type", title = "Distribution of Target Across Chest Pain Types") +
  theme_minimal()

cp1

#Resting Blood Pressure

rbps1 <- ggplot(data1, aes(x = factor(target), y = trestbps)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "Resting Blood Pressure", title = "Box Plot of Resting Blood Pressure by Target")

rbps1

#Age histogram of those with hypertension 
rbps2 <- ggplot(data = data1, aes(x = trestbps)) +
  geom_histogram(data = subset(data1, target == 1),
                 binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "RestingBPd", y = "Frequency", title = "Resting BPS Distribution for Target = 1")

rbps2

#Cholesterol

chol1 <- ggplot(data1, aes(x = factor(target), y = chol)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "Cholesterol", title = "Box Plot of Cholesterol by Target")

chol1

#Age histogram of those with hypertension 
chol2 <- ggplot(data = data1, aes(x = chol)) +
  geom_histogram(data = subset(data1, target == 1),
                 binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "Cholesterol", y = "Frequency", title = "Resting BPS Distribution for Target = 1")

chol2

#Fasting Blood Sugar
fbs1 <- ggplot(data1, aes(x = factor(target), fill = factor(fbs))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "Fasting Blood Sugar > 120 mg/dl", title = "Relationship of Target and Fasting Blood Sugar") +
  theme_minimal()

fbs1

#Resting ECG
recg1 <- ggplot(data1, aes(x = factor(target), fill = factor(restecg))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon", "2"="limegreen")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "Resting ECG", title = "Relationship of Target and Resting ECG") +
  theme_minimal()

recg1

#Maximum heartrate achieved
thalach1 <- ggplot(data1, aes(x = factor(target), y = thalach)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "Maximum Heart Rate", title = "Box Plot of Maximum Heart Rate by Target")

thalach1

#histogram of those with hypertension 
thalach2 <- ggplot(data = data1, aes(x = thalach)) +
  geom_histogram(data = subset(data1, target == 1),
                 binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "Maximum heart rate", y = "Frequency", title = "Maximum Heart Rate Distribution for Target = 1")

thalach2


#Exercise induced Angina
exang1 <- ggplot(data1, aes(x = factor(target), fill = factor(exang))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "Exercise Induced Angina", title = "Relationship of Target and Exercise Induced Angina") +
  theme_minimal()

exang1

#Oldpeak ST depression induced by exercise relative to rest.
oldpeak1 <- ggplot(data1, aes(x = factor(target), y = oldpeak)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "OldPeak", title = "Box Plot of OldPeak by Target")

oldpeak1

#histogram of those with hypertension 
oldpeak2 <- ggplot(data = data1, aes(x = oldpeak)) +
  geom_histogram(data = subset(data1, target == 1),
                 binwidth = 5, fill = "skyblue", color = "black") +
  labs(x = "Oldpeak", y = "Frequency", title = "Oldpeak for Target = 1")

oldpeak2

#Slope
slope1 <- ggplot(data1, aes(x = factor(target), fill = factor(slope))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon", "2"="limegreen")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "
       Slope", title = "Relationship of Target and Slope of Peak Exercise ST") +
  theme_minimal()

slope1

#ca Number of major vessels (0–3) colored by flourosopy
ca1 <- ggplot(data1, aes(x = factor(target), y = ca)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "ca", title = "Box Plot of ca by Target")

ca1

#Thal

thal1 <- ggplot(data1, aes(x = factor(target), fill = factor(thal))) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "skyblue", "1" = "salmon", "2"="limegreen", "3"="grey")) +  # Custom colors for target
  labs(x = "Target", y = "Count", fill = "
       Slope", title = "Relationship of Thal") +
  theme_minimal()

thal1

thal2 <- ggplot(data1, aes(x = factor(target), y = thal)) +
  geom_boxplot() +
  #eom_jitter(position = position_jitter(width = 0.1), alpha = 0.5)
  labs(x = "Target", y = "Thal", title = "Box Plot of Thal by Target")

thal2





# ---------------------- 2.2 Chi-Squared Test  --------------------------------
#Create df to store results
results.chisq.df <- data.frame(variable = character(), 
                               p_value = numeric(), 
                               stringsAsFactors = FALSE)

categorical_var_names <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "target") #Adding categorical variables by hand (since some categorical variables are in the num/int data form)

chisq_data <- cbind(data1[, categorical_var_names], target = data1$target)

for (var in categorical_var_names) {
  # Skip the 'target' variable itself for the Chi-squared test
  if(var != "target") {
    # Perform the Chi-squared test
    chisq_test_result <- chisq.test(table(chisq_data[[var]], chisq_data[["target"]]))
    
    # Create a new row for the results.chisq.df
    new_row <- data.frame(variable = var, 
                          p_value = chisq_test_result$p.value, 
                          stringsAsFactors = FALSE)
    
    # Append the new row to the existing results.chisq.df
    results.chisq.df <- rbind(results.chisq.df, new_row)
  }
}

# Print the results dataframe
print(results.chisq.df)

results.chisq.df[results.chisq.df$p_value <= 0.05,]

#Results of chi-sqaure test:
# variable       p_value
# 2       cp  0.000000e+00
# 3      fbs  4.291796e-08
# 4  restecg 1.986034e-183
# 5    exang  0.000000e+00
# 6    slope  0.000000e+00
# 7     thal  0.000000e+00





# ---------------------- 2.3 Kruskal Test  --------------------------------

results.kruskal.df <- data.frame(variable = character(), 
                                 p_value = numeric(), 
                                 stringsAsFactors = FALSE)


continuous_var_names <- c("age", "trestbps", "chol", "thalach", "oldpeak", "ca")  

kruskal_data <- cbind(data1[, continuous_var_names], target = data1$target)

kruskal_data$target <- factor(kruskal_data$target)

for(var_name in continuous_var_names) {
  # Perform the Kruskal-Wallis test between the continuous variable and the binary target
  test_result <- kruskal.test(reformulate('target', response = var_name), data = kruskal_data)
  
  # Create a new row for the results.kruskal.df
  new_row <- data.frame(variable = var_name, 
                        p_value = test_result$p.value, 
                        stringsAsFactors = FALSE)  # Ensure consistent data types
  
  # Append the new row to the existing results.kruskal.df
  results.kruskal.df <- rbind(results.kruskal.df, new_row)
}

# View the updated results dataframe
results.kruskal.df

# Return only variables where p-value <= 0.05
results.kruskal.df[results.kruskal.df$p_value <= 0.05,]

#From Kruskal: values with p <= 0.05 

# variable      p_value
# 1      age 2.277415e-04
# 2 trestbps 2.403466e-90
# 3     chol 3.758626e-78
# 4  thalach 0.000000e+00
# 5  oldpeak 0.000000e+00
# 6       ca 0.000000e+00

# ---------------------- 2.4 Selecting Variables  ------------------------------

#Results of chi-sqaure test: Variables with p <= 0.05
# variable       p_value
# 2       cp  0.000000e+00
# 3      fbs  4.291796e-08
# 4  restecg 1.986034e-183
# 5    exang  0.000000e+00
# 6    slope  0.000000e+00
# 7     thal  0.000000e+00

#From Kruskal: variables with p <= 0.05 

# variable      p_value
# 1      age 2.277415e-04
# 2 trestbps 2.403466e-90
# 3     chol 3.758626e-78
# 4  thalach 0.000000e+00
# 5  oldpeak 0.000000e+00
# 6       ca 0.000000e+00

final_data <- subset(data1, select = -sex)





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
# from testing session conducted on 26-03-24, 17000HRS.





# ==============================================================================
# ---------------------- 3.1 - train-test split  ------------------------------
#Data Preprocissing

#Converting categoricals into factor
categorical_var_names <- c("cp", "fbs", "restecg", "exang", "slope", "thal", "target")

# Convert each categorical variable in the dataset to a factor
for (var_name in categorical_var_names) {
  final_data[[var_name]] <- as.factor(final_data[[var_name]])
}

#Creating a 70-30 train-test split
traintest_split <- sample.split(Y = final_data$target, 
                                SplitRatio = 0.7)
ht_trainset <- subset(x = final_data, subset = traintest_split == TRUE)
ht_testset <- subset(x = final_data, subset = traintest_split == FALSE)


# Set the threshold
threshold <- 0.5


# Create empty data frame to store test set results for models
testset.model.compare <- data.frame('Model' = c("Logistic Regression with BE", 
                                                "CART", "Random Forest"),
                                    'Accuracy' = rep(0, 3), 'AUC' = rep(0, 3),
                                    'Precision' = rep(0, 3), 'Recall' = rep(0, 3), 
                                    'F1 Score' = rep(0, 3))

# ---------- 3.2 Logistic Regression (w/ backwards elimination)-----------------

#Creating the Log Model
log.model <- step(glm(target ~ ., 
                      family = binomial, 
                      data = ht_trainset), direction = 'backward')


# Important variables
summary(log.model)

#Results

# 10 Variables used (seen below)

# Call:
#   glm(formula = target ~ cp + trestbps + chol + restecg + thalach + 
#         exang + oldpeak + slope + ca + thal, family = binomial, data = ht_trainset)
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -0.1467669  0.3623463  -0.405   0.6854    
# cp1          1.0837496  0.0703093  15.414  < 2e-16 ***
#   cp2          1.7863843  0.0585701  30.500  < 2e-16 ***
#   cp3          1.7582767  0.0819984  21.443  < 2e-16 ***
#   trestbps    -0.0132420  0.0013036 -10.158  < 2e-16 ***
#   chol        -0.0009956  0.0004426  -2.250   0.0245 *  
#   restecg1     0.7164361  0.0472484  15.163  < 2e-16 ***
#   restecg2     0.3960842  0.2657173   1.491   0.1361    
# thalach      0.0122967  0.0012119  10.146  < 2e-16 ***
#   exang1      -0.7286971  0.0528307 -13.793  < 2e-16 ***
#   oldpeak     -0.5489929  0.0291952 -18.804  < 2e-16 ***
#   slope1      -0.7655183  0.1103481  -6.937 4.00e-12 ***
#   slope2      -0.0847789  0.1204256  -0.704   0.4814    
# ca          -0.8172408  0.0257852 -31.694  < 2e-16 ***
#   thal1        1.1815774  0.2695439   4.384 1.17e-05 ***
#   thal2        1.8250678  0.2578951   7.077 1.48e-12 ***
#   thal3       -0.0688797  0.2581432  -0.267   0.7896    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Testing accuracy on trainset

ht_trainset$target <- factor(ifelse(ht_trainset$target == 1, "yes", "no"))

trainset.prob <- predict(log.model, type = 'response')
trainset.predict <- ifelse(trainset.prob > threshold, "yes", "no")
trainset.predict <- factor(trainset.predict, levels = c("no", "yes"))

# Now, create the confusion matrix
ht_trainset_conf_matrix <- table(Actual = ht_trainset$target, Predicted = trainset.predict)

ht_trainset_conf_matrix
# Confusion matrix Results
#         Predicted
# Actual   no  yes
# no  6596 1653
# yes 1059 8933

#Further Analysis
library(caret)
log.trainset.cm <- confusionMatrix(ht_trainset_conf_matrix)
log.trainset.cm

# Accuracy : 0.8513          
# 95% CI : (0.8461, 0.8565)
# No Information Rate : 0.5803          
# P-Value [Acc > NIR] : < 2.2e-16  

#Testing accuracy on testset

#Converting testset target data into factor, with "yes" and "no" (For readability)
ht_testset$target <- factor(ifelse(ht_testset$target == 1, "yes", "no"))

# Evaluating model on testset - ACCURACY
testset.prob <- predict(log.model, type = 'response', newdata = ht_testset)
head(testset.prob)
# Comparing predicted probability of default with threshold
testset.predict <- ifelse(testset.prob > threshold, "yes", "no") 
head(testset.predict)
testset.predict <- factor(testset.predict, levels = c("no", "yes"))

ht_testset_conf_matrix <- table(Actual = ht_testset$target, Predicted = testset.predict)

ht_testset_conf_matrix

# Results
# Predicted
# Actual   no  yes
# no  2862  673
# yes  479 3803

mean(testset.predict == ht_testset$target) ## Overall accuracy = 0.8526289

testset.model.compare[1,2] <- mean(testset.predict == ht_testset$target)

#Evaluation of model

# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- ht_testset_conf_matrix[1,1]
FP <- ht_testset_conf_matrix[1,2]
FN <- ht_testset_conf_matrix[2,1]
TN <- ht_testset_conf_matrix[2,2]


# Evaluating model on test set - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[1,4] <- precision
testset.model.compare[1,5] <- recall
testset.model.compare[1,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC
pred = prediction(testset.prob, ht_testset$target)
auc = performance(pred, measure = "auc")
testset.model.compare[1,3] <- auc@y.values


# Evaluating model on test set - plot AUC
testset.predict.roc <- ifelse(testset.prob > threshold, 1, 0) # Comparing predicted probability of default with threshold
roc_score = roc(ht_testset$target, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")


# Q-Q Plot
par(mfrow= c(2,2))
plot(log.model)
par(mfrow= c(1,1))

# --------------------- 3.2 CART ------------------------------

# Training model on train set
cart.model <- rpart(target ~ . , data = ht_trainset, method = 'class',
                    control = rpart.control(minsplit = 2, cp = 0))

# to grow max tree, minsplit = 2 and cp = 0


# Plot maximal tree
rpart.plot(cart.model, nn=T, main = "Maximal Tree For Trainset")


# Display pruning sequence and 10-fold CV errors as a chart
plotcp(cart.model, main ="Subtrees in default")


# Display pruning sequence and 10-fold CV errors as a table
printcp(cart.model)

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was cp = 0.03157959.

# Classification tree:
#   rpart(formula = target ~ ., data = ht_trainset, method = "class", 
#         control = rpart.control(minsplit = 2, cp = 0))
# 
# Variables actually used in tree construction:
#   [1] ca       chol     cp       exang    oldpeak  slope    thal     thalach  trestbps
# 
# Root node error: 8249/18241 = 0.45222
# 
# n= 18241 
# 
# CP nsplit rel error    xerror       xstd
# 1  0.4920596      0 1.0000000 1.0000000 0.00814894
# 2  0.0484301      1 0.5079404 0.5079404 0.00688708
# 3  0.0401261      3 0.4110801 0.4110801 0.00636944
# 4  0.0141229      5 0.3308280 0.3360407 0.00587763
# 5  0.0115772      8 0.2873076 0.2989453 0.00559829
# 6  0.0095769     10 0.2641532 0.2654867 0.00532166
# 7  0.0081222     13 0.2354225 0.2474239 0.00516123
# 8  0.0078070     15 0.2191781 0.2283913 0.00498272
# 9  0.0073948     20 0.1801430 0.1870530 0.00455606
# 10 0.0072736     21 0.1727482 0.1745666 0.00441492
# 11 0.0071524     22 0.1654746 0.1642623 0.00429346
# 12 0.0070312     24 0.1511698 0.1475330 0.00408555
# 13 0.0069705     25 0.1441387 0.1344405 0.00391241
# 14 0.0069503     28 0.1230452 0.1344405 0.00391241
# 15 0.0065462     31 0.1021942 0.1179537 0.00367919
# 16 0.0064250     33 0.0891017 0.1008607 0.00341604
# 17 0.0055764     35 0.0762517 0.0820706 0.00309514
# 18 0.0040005     36 0.0706752 0.0756455 0.00297600
# 19 0.0038186     38 0.0626743 0.0711601 0.00288945
# 20 0.0034550     40 0.0550370 0.0618257 0.00269914
# 21 0.0028286     42 0.0481270 0.0383077 0.00213623
# 22 0.0025862     46 0.0339435 0.0229119 0.00165794
# 23 0.0023033     54 0.0046066 0.0066675 0.00089769
# 24 0.0000000     56 0.0000000 0.0000000 0.00000000

#Using Cross-Validation to identify optimal CP

trainControl <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation
cv.model <- train(target ~ ., data = ht_trainset, 
                  method = "rpart",  # For CART models; change if using different models
                  trControl = trainControl)

print(cv.model)

#Results
# cp          Accuracy   Kappa    
# 0.03157959  0.8226535  0.6385058
# 0.04145957  0.7892128  0.5694854
# 0.49205964  0.6332794  0.2073696

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was cp = 0.03157959.

# Prune the max tree
prune.cart.model <- prune(cart.model, cp = 0.03157959)
print(prune.cart.model)
rpart.plot(prune.cart.model, nn=T, main = "Optimal Tree for Trainset")

# 4 Variables used: thal, cp, ca, slope

# Accuracy of model on train set
cart.yhat <- predict(prune.cart.model, type= 'class')
cm.trainset <- table(Actual = ht_trainset$target ,
                     Predicted = cart.yhat, deparse.level = 2)
cm.trainset
mean(cart.yhat == ht_trainset$target) ## Overall Accuracy = 0.850392

#Accuracy of Model on testset
prune.cart.model.yhat <- predict(prune.cart.model, type= 'class', newdata = ht_testset)
cm.testset <- table(Actual = ht_testset$target , 
                    Predicted = prune.cart.model.yhat, deparse.level = 2)
cm.testset
mean(prune.cart.model.yhat == ht_testset$target) ## Overall accuracy = 0.8560829
testset.model.compare[2,2] <- mean(prune.cart.model.yhat == ht_testset$target)

# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- cm.testset[1,1]
FP <- cm.testset[1,2]
FN <- cm.testset[2,1]
TN <- cm.testset[2,2]


# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[2,4] <- precision
testset.model.compare[2,5] <- recall
testset.model.compare[2,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC 
testset.prob <- as.data.frame(predict(prune.cart.model, type= 'prob', newdata = ht_testset))
pred = prediction(testset.prob$yes, ht_testset$target)
auc = performance(pred, measure = "auc")
testset.model.compare[2,3] <- auc@y.values


# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(testset.prob$yes > threshold, 1, 0)
roc_score = roc(ht_testset$target, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")



# ------------------------ 3.3 Random Forest -----------------------------------

# Training model on train set
rf.model <- randomForest(target ~ . , data = ht_trainset, 
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model on train set
rf.model

# OOB estimate of  error rate: 0%
# Confusion matrix:
#   no  yes class.error
# no  8249    0           0
# yes    0 9992           0

# 0% Error suggests the model may be overfitted to the trainset

# Evaluating OOB Error Rate
plot(rf.model,
     main = "OOB Error Rates Of Random Forest On Dyslipidemia Dataset
     Up Till 500 Trees") #Can add this to the report

## OOB error rate stabilised before 50 trees

# Evaluating model on test set - accuracy

test_predictions <- predict(rf.model, newdata = ht_testset)
accuracy <- sum(test_predictions == ht_testset$target) / nrow(ht_testset)
print(accuracy)

#Accuracy = 1

confusion_matrix <- table(Predicted = test_predictions, Actual = ht_testset$target)
print(confusion_matrix)

#Confusion Matrix
#            Actual
# Predicted   no  yes
#       no  3535    0
#       yes    0 4282

importance(rf.model)
varImpPlot(rf.model)

#Results
#                no       yes MeanDecreaseAccuracy MeanDecreaseGini
# age       1.96580  3.041974             3.382378         8.179202
# cp       61.52970 62.467949            67.682549      1371.583359
# trestbps 62.77013 55.372573            64.572161       727.354659
# chol     64.70406 63.023593            68.106289       878.176192
# fbs      24.77421 22.787861            26.493130        73.496708
# restecg  32.56952 33.580546            34.536433       206.941071
# thalach  52.20576 61.262552            61.647631      1113.291474
# exang    36.30538 32.973338            37.231493       501.307918
# oldpeak  60.17220 53.305310            64.593639      1043.895298
# slope    38.10736 37.651037            40.844427       437.794932
# ca       63.91474 64.835668            68.807696      1110.825375
# thal     57.36424 61.357697            63.619335      1387.760649

# From MeanDecreaseAccuracy
# cp,  chol, and ca (67.682549,68.106289 and 68.807696) seem particularly important, suggesting these features are significant predictors 

#10-fold Cross-Validation (Due to Accuracy = 1 on both trainset and testset)

train_control <- trainControl(method = "cv", number = 10, savePredictions = "final", classProbs = TRUE)

rf_cv_model <- train(target ~ ., data = ht_trainset, method = "rf",
                     trControl = train_control, metric = "Accuracy",
                     tuneLength = 5)

# Print the model results
print(rf_cv_model)

#Results
# Random Forest 
# 
# 18241 samples
# 12 predictor
# 2 classes: 'no', 'yes' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 16416, 16417, 16417, 16417, 16417, 16417, ... 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy   Kappa    
# 2    0.9765908  0.9525968
# 6    1.0000000  1.0000000
# 10    1.0000000  1.0000000
# 14    1.0000000  1.0000000
# 18    1.0000000  1.0000000
# 
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 6.

# Summary of results
results <- rf_cv_model$results
print(results)

# mtry  Accuracy     Kappa AccuracySD     KappaSD
# 1    2 0.9765908 0.9525968 0.00320048 0.006503697
# 2    6 1.0000000 1.0000000 0.00000000 0.000000000
# 3   10 1.0000000 1.0000000 0.00000000 0.000000000
# 4   14 1.0000000 1.0000000 0.00000000 0.000000000
# 5   18 1.0000000 1.0000000 0.00000000 0.000000000

#Implementing the Random forest with mtry = 6

optimal_rf_model <- randomForest(target ~ ., data = ht_trainset,
                                 na.action = na.omit, 
                                 mtry = 6,  # Set based on cross-validation results
                                 ntree = 50,  # Where OOB values stabilize
                                 importance = TRUE)

optimal_rf_model

#Results
# Call:
#   randomForest(formula = target ~ ., data = ht_trainset, mtry = 6,      ntree = 50, importance = TRUE, na.action = na.omit) 
# Type of random forest: classification
# Number of trees: 50
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 0%
# Confusion matrix:
#   no  yes class.error
# no  8249    0           0
# yes    0 9992           0

#Testing accuracy on trainset
rf.model.yhat <- predict(optimal_rf_model, newdata = ht_testset)
ht.rf.cm.testset <- table(Actual = ht_testset$target ,
                    Predicted = rf.model.yhat)
ht.rf.cm.testset
mean(rf.model.yhat == ht_testset$target) ## Overall accuracy = 1
testset.model.compare[3,2] <- mean(rf.model.yhat == ht_testset$target)

# Assign True Positive (TP) / True Negative (TN) / False Positive (FP) / False Negative (FN)
TP <- ht.rf.cm.testset[1,1]
FP <- ht.rf.cm.testset[1,2]
FN <- ht.rf.cm.testset[2,1]
TN <- ht.rf.cm.testset[2,2]

# Evaluating model on testset - precision and recall
precision = (TP) / (TP + FP)
recall = (TP) / (TP + FN)
testset.model.compare[3,4] <- precision
testset.model.compare[3,5] <- recall
testset.model.compare[3,6] <- (2 * precision * recall) / (precision + recall)


# Evaluating model on test set - AUC
ht_testset.prob <- as.data.frame(predict(rf.model, type = 'prob', newdata = ht_testset))
pred = prediction(ht_testset.prob$yes, ht_testset$target)
auc = performance(pred, measure = "auc")
testset.model.compare[3,3] <- auc@y.values

# Evaluating model on testset - plot AUC
testset.predict.roc <- ifelse(ht_testset.prob$yes > threshold, 1, 0) # Comparing predicted probability of default with threshold
roc_score = roc(ht_testset$target, testset.predict.roc)
plot(roc_score, main = "AUC for Logistic Regression w BE")

# Evaluation of models
testset.model.compare

#Comparison

#                           Model  Accuracy       AUC Precision    Recall  F1.Score
# 1 Logistic Regression with BE 0.8526289 0.9260502 0.8096181 0.8566298 0.8324607
# 2                        CART 0.8560829 0.8831762 0.8475248 0.8364042 0.8419278
# 3               Random Forest 1.0000000 1.0000000 1.0000000 1.0000000 1.0000000

# -------------------- 3.3.2 Random Forest (All data) --------------------------
# Apply RF on entire final_data dataset
rf.model <- randomForest(target ~ . , data = final_data,
                         na.action = na.omit,
                         importance = TRUE,
                         keep.inbag = TRUE)


# Accuracy of model
rf.model

