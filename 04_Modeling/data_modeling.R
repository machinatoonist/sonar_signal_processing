library(AppliedPredictiveModeling)
library(tidyverse)
library(mlbench)
library(forcats)
library(caret)
library(caTools)
library(h2o)
h2o.init()

# install.packages("e1071")
# install.packages("caTools")

# Import dataset
# Analysing sonar data was one of the first applications of machine learning
# Differentiating between a rock "R" and a mine "M" using 60 different sonar signal characteristics
data("Sonar")

glimpse(Sonar)

Sonar[1:6,c(1:6,61)]

# Understand the number of levels being classifed
levels(Sonar$Class)

# Randomlse the dataset
set.seed(42) # Use set.seed() so that results can be reproduced and compared
rows <- sample(nrow(Sonar))
Sonar_tbl <- Sonar[rows,]
glimpse(Sonar_tbl)

# Manual split of data 60:40 rather than 80:20 for training and testing
split <- round(nrow(Sonar_tbl)*0.6)
sonar_train_tbl <- Sonar_tbl[1:split,]
sonar_test_tbl <- Sonar_tbl[(split + 1):nrow(Sonar_tbl),]

glimpse(sonar_test_tbl)
glimpse(sonar_train_tbl)

# Confirm train set size
nrow(sonar_train_tbl)/nrow(Sonar_tbl)


# Create trainControl object: myControl

# Fit glm model: model
model <- glm(
  Class ~ ., 
  family = "binomial",
  sonar_train_tbl
)

# Fit h2o AutoML
y <- "Class"
x <- setdiff(names(sonar_train_tbl), y)

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame = as.h2o(sonar_train_tbl),
  # validation_frame = as.h2o(sonar_test_tbl),
  # leaderboard_frame = test_h2o,
  max_runtime_secs = 30,
  seed = 1,
  max_models = 20
  # nfolds = 5
)

# Inspect the h2o results object

typeof(automl_models_h2o)

slotNames(automl_models_h2o)

automl_models_h2o@leaderboard

automl_models_h2o@leader

automl_models_h2o@leaderboard %>%
  as_tibble() %>% 
  slice(1:6) %>%
  pull(model_id) %>%
  h2o.getModel()

extract_model_name_by_position <- function (h2o_leaderboard, n = 1, verbose = TRUE){
  
  model_name <- h2o_leaderboard %>%
    as_tibble() %>% 
    slice(n) %>%
    pull(model_id) 
  
  if(verbose) message(model_name)
  
  return(model_name)
}

# Save Top 6 Models
for(i in 1:6) {
  automl_models_h2o@leaderboard %>% 
    extract_model_name_by_position(i) %>%
    h2o.getModel() %>%
    h2o.saveModel(path = "04_Modeling/h2o_models/")
}


# Single model
sonar_h2o_ensemble <- h2o.loadModel(path = "04_Modeling/h2o_models/StackedEnsemble_AllModels_AutoML_20210113_130138")

p_h2o <- h2o.predict(sonar_h2o_ensemble,  newdata = as.h2o(sonar_test_tbl)) %>% as_tibble()



# Predict on test: p
p <- predict(model, sonar_test_tbl, type = "response") 

# Store results and convert prediction to a factor
test_results <- sonar_test_tbl %>%
  select(Class) %>%
  mutate(probability = p,
         pred_class = if_else(probability > 0.5, "M", "R") %>% 
           as_factor()) 

# Create confusion matrix
confusionMatrix(test_results$pred_class, test_results$Class)

# A 50% threshold may not be a great choice for this application.
# The cost of a false negative is likely to be far greater than a false positive
# For classification problems a Receiver Operator Characteristic (ROC) Curve
# This illustrates how the diagnositic ability of a binary classifier system 
# changes as its discrimination threshold is varied.

# Make ROC curve using caTools() package
colAUC(test_results$probability, test_results$Class, plotROC = TRUE)

# ROC curve for h2o model
colAUC(p_h2o$M, sonar_test_tbl$Class, plotROC = TRUE) 

myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

