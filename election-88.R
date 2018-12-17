# https://greta-stats.org/articles/analyses/election_88.html#session-information

library(tidyverse)
library(bayesplot)
library(greta)

theme_set(theme_bw())

# gather data

root <- "https://raw.githubusercontent.com/stan-dev/example-models/master"
model_data <- "ARM/Ch.14/election88_full.data.R"
source(file.path(root, model_data))

# What effect did race and gender have on voting outcomes in 1988?

# Exploratory analysis

dev.new()
ggplot(data.frame(y)) +
    geom_bar(aes(y)) + 
    ggtitle("Distribution of voting outcomes")

ggplot(data.frame(female)) +
    geom_bar(aes(female)) + 
    ggtitle("Distribution of female indicator")

ggplot(data.frame(black)) +
    geom_bar(aes(black)) + 
    ggtitle("Distribution of black indicator")

ggplot(data.frame(state)) +
    geom_bar(aes(state)) + 
    ggtitle("Distribution of values in state")

state_recoded <- dense_rank(state)
table(state_recoded)

# Model

n <- length(y)
n_states <- max(state_recoded)

# Define the data objects to be used in the model
y_greta <- as_data(y)
black_greta <- as_data(black)
female_greta <- as_data(female)
state_greta <- as_data(state_recoded)



