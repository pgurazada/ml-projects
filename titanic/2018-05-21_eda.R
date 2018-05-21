#' ---
#' title: "Exploratory analysis of the titanic data set"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon May 21 06:17:06 2018

library(tidyverse)
library(caret)

set.seed(20130810)

theme_set(theme_bw())

dev.new()

train_df <- read_csv('titanic/data/train.csv', progress = TRUE)

glimpse(train_df)

#' Brief summary of the npon-obvious features:
#' - `Pclass`: class of the ticket
#' - `SibSp`: Number of siblings and spouses aboard
#' - `Parch`: Number of parents and children aboard
#' - `Ticket`: Ticket number
#' - `Embark`: port of embarkation

#' Lets see where the missing values are concentrated

is.na(train_df) %>% 
  colSums() %>% 
  sort(., decreasing = TRUE) * 100/nrow(train_df)

#' We might have to take out `Cabin` since most of it is missing

ggplot(train_df) +
  geom_histogram(aes(x = Survived), bins = 3)

ggplot(train_df) +
  geom_bar(aes(x = factor(Survived), fill = factor(Pclass)), width = 0.4) +
  scale_fill_grey('Passenger class') +
  scale_x_discrete('Survival status', labels = c('Not survived', 'survived')) + 
  coord_flip()
