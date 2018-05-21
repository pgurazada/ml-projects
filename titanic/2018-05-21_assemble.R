#' ---
#' title: "Assembe the training data by selectig/transforming features from the raw data"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon May 21 10:49:45 2018

#' In this script, we munge the raw data, perform initial extraction of features
#' and assemble the data. In an normal scenario, this processed data would then 
#' be fed into scikit-learn to make a train-test split and model exploration

library(tidyverse)

train_df <- read_csv('titanic/data/train.csv', progress = TRUE)

glimpse(train_df)

train_df %>% 
  select(-PassengerId, -Name, -Ticket, -Cabin) %>% 
  mutate(log_fare = log(Fare+0.001)) %>% # This is to handle the 0 fare getting bumped up to NA
  select(-Fare) ->
  train_df  

#' Quick check on the NA's (to see that we did not induce new ones)

colSums(is.na(train_df))

write_csv(train_df, 'titanic/processed/titanic_train.csv')

