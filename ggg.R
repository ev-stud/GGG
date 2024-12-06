library(tidymodels)
library(vroom)

trainggg <- vroom("./GGG/train.csv")
testggg <- vroom("./GGG/test.csv")
trainmissing <- vroom("./GGG/trainWithMissingValues.csv")


# Impute Missing Data -----------------------------------------------------
impute_recipe <- recipe(type~., trainggg) %>%
  step_mutate_at(c(color,type), fn= factor) %>%
  step_impute_knn(c(bone_length,rotting_flesh,hair_length), impute_with =imp_vars(all_predictors()), neighbors=3) 
  # > imp_vars() are the variables used to fill in the specified missing variables


imputedSet <- bake(prep(ggg_recipe), trainmissing)

# did it impute missing data?
sum(is.na(trainmissing$bone_length))
sum(is.na(imputedSet$bone_length)) # yes!

# compute the RMSE of the imputed data
rmse_vec(trainggg[is.na(trainmissing)],
         imputedSet[is.na(trainmissing)])



# SVM ---------------------------------------------------------------------
library(embed)

trainggg$type <- as.factor(trainggg$type)

svm_recipe <- recipe(type~., trainggg) %>%
  #update_role(id, new_role="id variable") %>%
  step_rm(id) %>%
  step_mutate_at(color, fn= factor)

svmLinear <- svm_linear(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly <- svm_poly(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial <- svm_rbf(cost=tune()) %>% # set or tune cost penalty
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svmPoly)

tuning_grid <- grid_regular(cost(),
                            levels = 3) # grid of L^2 tuning possibilities

folds <- vfold_cv(trainggg, v = 5, repeats =2) # K-folds

cv_results <- svm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc") # or tune roc_auc

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=trainggg)

svm_submit <- final_wf %>% 
  predict(new_data= testggg, type="class") %>% # classifies result based on highest prob
  bind_cols(testggg) %>%
  rename(type=.pred_class) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,type)

vroom_write(svm_submit,"./GGG/gggSVM.csv", delim = ",")

### Naive Bayes
library(discrim) # for naivebayes model
library(embed)
library(themis) # smote
library(naivebayes) # for multinomials

my_recipe <- recipe(type~., trainggg) %>%
  #step_rm(id) %>%
  #update_role(id, new_role="id variable") %>%
  step_mutate(id, features = id) %>%
  step_mutate_at(color, fn= factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize(all_predictors()) %>%
  #step_pca(all_predictors(), threshold = 0.9) %>%
  step_bsmote(all_outcomes(), neighbors=2)

# multi_ not setup for recipes? multinomial_naive_bayes
nb_model <- naive_Bayes(Laplace = tune(), # parameter that weights prior vs. posterior probs.
                        smoothness=tune()) %>% #density smoothness, ie. bin width for datapoints
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# tune parameters
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 12) # grid of L^2 tuning possibilities

folds <- vfold_cv(trainggg, v = 15, repeats =2) # K-folds

cv_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="accuracy")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=trainggg)

nb_submit <- final_wf %>% 
  predict(new_data= testggg, type="class") %>%
  bind_cols(testggg) %>%
  rename(type=.pred_class) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,type)

vroom_write(nb_submit,"./GGG/gggNB.csv", delim = ",")

### Neural Networks
library(remotes) 
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
keras::install_keras()
# or use BYU servers instead of ^

nn_recipe <- recipe(type~., trainggg) %>%
  update_role(id, new_role="id") %>% # id column should not be included as a category
  step_mutate_at(color, fn= factor) %>% 
  step_dummy(color) %>% # neural networks cannot include categorical variable
  step_range(all_numeric_predictors(), min=0, max=1) # Xs must be in [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>% # num of iterations before completing a hidden layer transformation
  set_engine("keras") %>% # verbose = 0 prints off less
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tunegrid <- grid_regular(hidden_units(range=c(1, 20)), # determine max range, or finalize it
                            levels = 10) # num of variables per hidden layer?

folds <- vfold_cv(trainggg, v = 5, repeats =1) # K-folds

cv_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=nn_tunegrid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="accuracy") # or tune roc_auc

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=trainggg)

nn_submit <- final_wf %>% 
  predict(new_data= testggg, type="class") %>% # classifies result based on highest prob
  bind_cols(testggg) %>%
  rename(type=.pred_class) #%>% # pred_1 is prediction on response = 1, pred_0 for response=0
  
nn_submit[,c(2,1)]

vroom_write(nn_submit[,c(2,1)],"./GGG/gggNN.csv", delim = ",")


### Boosted Trees & BART
library(tidymodels)
library(bonsai)
library(lightgbm)
library(dbarts)
library(BART) # for multinomial categorical response

my_recipe <- recipe(type~., trainggg) %>%
  step_rm(id) %>%
  step_mutate_at(color, fn= factor) %>% 
  step_lencode_glm(all_factor_predictors(), outcome = vars(type)) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors=2)

boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% # or xgboost, but its slower
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

boost_tunegrid <- grid_regular(tree_depth(), # determine max range, or finalize it
                           trees(),
                           learn_rate(), # a penalized contribution of each tree to the overall model
                            levels = 3) 

folds <- vfold_cv(trainggg, v = 5, repeats =1) # K-folds

cv_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=boost_tunegrid,
            metrics=NULL)

# OR #
bart_model <- BART::mbart(trees=tune()) %>% # multinomial categorical response
  # parsnip::bart(trees=tune()) %>%
  # BART figures out depth and learn_rate
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model) # or boost model

bart_tunegrid <- grid_regular(trees(), # a penalized contribution of each tree to the overall model
                           levels = 3) 

folds <- vfold_cv(trainggg, v = 5, repeats =1) # K-folds

cv_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=bart_tunegrid,
            metrics=NULL) #metric_set(roc_auc,f_meas,sens,recall,spec,precision,accuracy)

bestTune <- cv_results %>%
  select_best(metric="roc_auc") # or tune accuracy

final_wf <- b_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=trainggg)

b_submit <- final_wf %>% 
  predict(new_data= testggg, type="class") %>% # classifies result based on highest prob
  bind_cols(testggg) %>%
  rename(type=.pred_class) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,type)

vroom_write(b_submit,"./GGG/gggSubmit.csv", delim = ",")

### LDA
library(MASS) # lda engine
library(discrim) # lda
library(themis) # smote

my_recipe <- recipe(type~., trainggg) %>%
  #step_rm(id) %>%
  #update_role(id, new_role="id variable") %>%
  #step_mutate(id, features = id) %>%
  step_mutate_at(color, fn= factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_normalize(all_predictors()) %>%
  #step_pca(all_predictors(), threshold = 0.9) %>%
  step_bsmote(all_outcomes(), neighbors=2)

lda_model <- discrim_linear() %>% 
  set_engine("MASS")

tidy(my_recipe)

lda_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lda_model) %>%
  fit(data=trainggg)

lda_preds <- predict(lda_wf, new_data = testggg, 
                     type = "class") # type: classification/probability

lda_submit <- lda_preds %>%
  bind_cols(testggg) %>%
  rename(type=.pred_class) %>% # pred_1 is prediction on response = 1, pred_0 for response=0
  select(id,type) 

vroom_write(lda_submit,"./GGG/gggLDA.csv", delim = ",")

