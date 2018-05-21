#' ---
#' title: "Exploratory analysis of the titanic data set"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon May 21 10:22:30 2018

library(tidyverse)

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

#' We might have to take out `Cabin` since most of it is missing. There are
#' several categorical features so we can use barplots to understand how they
#' vary within survivors and non-survivors. Some features are numeric, so we
#' will check if their distribution is skewed across both classes in the target

ggplot(train_df) +
  geom_histogram(aes(x = Survived), bins = 3)

#' The above plot shows no signs of a bad class imbalance

ggplot(train_df) +
  geom_bar(aes(x = factor(Survived), fill = factor(Pclass)), width = 0.4) +
  scale_fill_grey('Passenger class') +
  scale_x_discrete('Survival status', labels = c('Not survived', 'survived')) + 
  coord_flip()

#' The above plot tells a lot. It seems that first class passengers were most
#' prominent among the survivors. Most third class passengers did not survive

ggplot(train_df) +
  geom_bar(aes(x = factor(Survived), fill = factor(Sex)), width = 0.4) +
  scale_fill_grey('Gender') +
  scale_x_discrete('Survival status', labels = c('Not survived', 'survived')) + 
  coord_flip()

#' Once again, female passengers dominate the survivors, while male passengers
#' are an overwhelming proportion of the non-survivors

ggplot(train_df) +
  geom_histogram(aes(x = Age, fill = factor(Survived)), bins =  20) +
  scale_fill_grey('Survived')

#' The age distribution among the survivors and non survivors seems to be of the
#' same shape. We would expect passengers of lower ages to have a higher
#' survival chance

ggplot(train_df) +
  geom_bar(aes(x = factor(SibSp), fill = factor(Survived)), position = 'dodge', width = 0.4) +
  scale_fill_grey('Survived') +
  scale_x_discrete('Number of siblings/spouses on board') +
  coord_flip()

#' Difficult to tell since the numbers are low, but survival rate among people
#' with more siblings/spouses on board seems to be high

ggplot(train_df) +
  geom_bar(aes(x = factor(Parch), fill = factor(Survived)), position = 'dodge', width = 0.4) +
  scale_fill_grey('Survived') +
  scale_x_discrete('Number of parents/children on board') +
  coord_flip()

#' Once again, the pattern is the same but the impact of having family members
#' on board is difficult to tell from the two plots above

ggplot(train_df) +
  geom_histogram(aes(x = Fare, fill = factor(Survived)), bins =  20) +
  scale_fill_grey('Survived')

#' Fare distribution is skewed among the survivors and non-survivors. A log
#' transformation might be advisable

ggplot(train_df) +
  geom_histogram(aes(x = log(Fare)), bins =  20) +
  scale_fill_grey('Survived')

#' log transform helps a bit but not too much. 
#' 
#' We can check if the warning from the log transform corresponds to the zero values
#' in `Fare`

all.equal(sum(train_df$Fare == 0), 15)

ggplot(train_df) +
  geom_bar(aes(x = factor(Survived), fill = factor(Embarked)), width = 0.4, position = 'dodge') +
  scale_fill_grey('Embarkation') +
  scale_x_discrete('Survival status', labels = c('Not survived', 'survived')) + 
  coord_flip()

#' Overall, apart from a possible log transformation of fare, there is no
#' specific feature engineering that seems to be needed


