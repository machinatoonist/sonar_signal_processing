# 1 Load libraries ----

library(AppliedPredictiveModeling)
library(tidyverse)
library(tidyquant)
library(mlbench)
library(forcats)
library(caret)
library(caTools)
library(fs)
library(glue)
library(h2o)
h2o.init()

wd = getwd()

# install.packages("e1071")
# install.packages("caTools")

# 2 Prepare dataset ----
# Analysing sonar data was one of the first applications of machine learning
# Differentiating between a rock "R" and a mine "M" using 60 different sonar signal characteristics
data("Sonar")
data("wine")

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


# 3 Fit model using caret package ----

# 3.1 Fit glm model: model ----
model <- glm(
  Class ~ ., 
  family = "binomial",
  sonar_train_tbl
)

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

# Make ROC curve using caTools() and caret package
colAUC(test_results$probability, test_results$Class, plotROC = TRUE)

# First glm model is not a good classifier.  Let's try to improve it.
# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,  # Required to use AUC to rank classification models
  classProbs = TRUE, # Must be true for classification models
  verboseIter = TRUE
)

model_caret_glm <- train(Class ~ ., data = sonar_train_tbl, method = "glm", trControl = myControl)
model_caret_glm
# plot(model_caret_glm) # no tuning parameters for glm

# From previous step
# tuneGrid <- data.frame(
#   .mtry = c(2, 3, 7),
#   .splitrule = "variance",
#   .min.node.size = 5
# )

# 3.2 Fit random forest: model ----
model_caret_rf <- train(
    Class ~.,
    tuneLength = 6,
    # tuneGrid = tuneGrid,
    data = sonar_train_tbl, 
    method = "ranger",
    trControl = myControl
    )

# Print model to console
model_caret_rf
plot(model_caret_rf)

# Predict on test: p
p <- predict(model_caret_rf, sonar_test_tbl, type = "prob") 

# Store results and convert prediction to a factor
test_results_rf <- sonar_test_tbl %>%
  select(Class) %>%
  mutate(probability = predict(model_caret_rf, sonar_test_tbl, type = "prob"))

# Make ROC curve using caTools() and caret package
colAUC(test_results_rf$probability, test_results_rf$Class, plotROC = TRUE)

# 3.3 Fit glmnet model ----
# Train glmnet with custom trainControl and tuning: model
model_caret_glmnet <- train(
  Class ~.,
  data = sonar_train_tbl,
  tuneGrid = expand.grid(
    alpha = seq(0,1,0.2),  # Lasso - alpha = 1, Ridge - alpha = 0
    lambda = seq(0.0001, 1, length = 100)
    ),
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model_caret_glmnet
plot(model_caret_glmnet)
# Predict on test: p
p <- predict(model_caret_glmnet, sonar_test_tbl, type = "prob") 

# Store results and convert prediction to a factor
test_results_glmnet <- sonar_test_tbl %>%
  select(Class) %>%
  mutate(probability = predict(model_caret_glmnet, sonar_test_tbl, type = "prob"))

# Make ROC curve using caTools() and caret package
colAUC(test_results_glmnet$probability, test_results_glmnet$Class, plotROC = TRUE)

# Print maximum ROC statistics
max(model_caret_glmnet[["results"]]$ROC)
max(model_caret_glm[["results"]]$ROC)
max(model_caret_rf[["results"]]$ROC)

# Save and load caret models

model_caret_rf

# Save caret models
saveRDS(model_caret_rf, glue("{wd}/04_Modeling/caret_models/model_caret_rf.rds"))
saveRDS(model_caret_glmnet, glue("{wd}/04_Modeling/caret_models/model_caret_glmnet.rds"))
saveRDS(model_caret_glm, glue("{wd}/04_Modeling/caret_models/model_caret_glm.rds"))

# Load caret models
model_caret_glm <- readRDS(glue("{wd}/04_Modeling/caret_models/model_caret_glm.rds"))
model_caret_glmnet <- readRDS(glue("{wd}/04_Modeling/caret_models/model_caret_glmnet.rds"))
model_caret_rf <- readRDS(glue("{wd}/04_Modeling/caret_models/model_caret_rf.rds"))

# 4.1 Fit h2o AutoML ----
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
    h2o.saveModel(path = glue("{wd}/04_Modeling/h2o_models/"))
}


# Single model
sonar_h2o_ensemble <- h2o.loadModel(path = glue("{wd}/04_Modeling/h2o_models/StackedEnsemble_AllModels_AutoML_20210113_130138"))

p_h2o <- h2o.predict(sonar_h2o_ensemble,  newdata = as.h2o(sonar_test_tbl)) %>% as_tibble()

# Quick ROC Plot
colAUC(p_h2o$M, sonar_test_tbl$Class, plotROC = TRUE) 


# 4.2 ROC curve for h2o model ----

performance_h2o <- h2o.performance(sonar_h2o_ensemble, newdata = as.h2o(sonar_test_tbl))
typeof(performance_h2o)
performance_h2o %>% slotNames()
performance_h2o@algorithm
performance_h2o@metrics


# Classifier Summary Metrics
h2o.auc(performance_h2o)
h2o.giniCoef(performance_h2o)
h2o.logloss(performance_h2o)

# Performance on training data
h2o.confusionMatrix(sonar_h2o_ensemble)

# Evaluate performance on the test set
h2o.confusionMatrix(performance_h2o)

# Precision vs Recall Plot

performance_tbl <- performance_h2o %>%
  h2o.metric() %>%
  as_tibble() 

performance_tbl %>%
  arrange(desc(f1)) %>%
  glimpse()

performance_tbl %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = precision, color = "orange")) +
  geom_line(aes(y = recall, color = "blue")) +
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) + 
  theme_tq() +
  labs(
    title = "Precision vs Recall",
    y = "value"
  )


# Function to load performance metrics for models
path <- glue("{wd}/04_Modeling/h2o_models/StackedEnsemble_AllModels_AutoML_20210113_130138")
test_tbl <- sonar_test_tbl

load_model_performance_metrics <- function (path, test_tbl) {
  
  model_h2o <- h2o.loadModel(path)
  perf_h2o <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl))
  
  perf_h2o %>%
    h2o.metric() %>% 
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc, precision, recall)
}

load_model_performance_metrics(path, sonar_test_tbl) 

# Get the path to each model and load performance metrics
model_metrics_tbl <- fs::dir_info(path = glue("{wd}/04_Modeling/h2o_models/"), fail = TRUE) %>%
  select(path) %>%
  mutate(metrics = map(path, load_model_performance_metrics, test_tbl)) %>%
  unnest(cols = c(metrics)) %>% filter(auc > 0.87)

number_of_folders_in_path <- 9
model_metrics_tbl %>%
  mutate(
    Prediction_Method = str_split(string = path, pattern = "/",simplify = T)[,number_of_folders_in_path] %>% as_factor(),
    AUC = auc %>% round(3) %>% as.character() %>% as_factor()) %>%
  ggplot(aes(x = fpr, y = tpr, color = Prediction_Method, linetype = AUC)) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1, color = "Random Guessing", linetype = "0.5"), size = 0.1) + 
  geom_line(size = .6) +
  theme_tq() +
  scale_color_tq() +
  theme(legend.direction = "vertical") +
  labs(
    title = "ROC Plot - Performance of Sonar Signal Processing Models",
    subtitle = "How to tell the difference between a rock and a mine from a sonar receiver signal?",
    caption = "The ROC curve was first developed by engineers during World War II. The closer the area under the curve (auc) is to 1 the better",
    x = "False positive rate",
    y = "True positive rate"
  )
