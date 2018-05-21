#' ---
#' title: "EDA for the wells data"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Mon May 21 14:14:10 2018
#' 

library(tidyverse)

set.seed(20130810)

dev.new()

theme_set(theme_bw())

wells_features <- read_csv('wells-africa/data/well_features.csv', 
                           progress = TRUE)

wells_labels <- read_csv('wells-africa/data/well_labels.csv',
                         progress = TRUE)

glimpse(wells_features)
glimpse(wells_labels)

#' The labels are strings and hence can be fixed once we agree on a naming scheme

wells_labels %>% 
  mutate(status = case_when(status_group == "functional" ~ 1,
                            status_group == "non functional" ~ 2,
                            TRUE ~ 0)) ->
  wells_labels

#' Lets see if there are any missing values

wells_features %>% 
  is.na() %>% 
  colSums() %>% 
  sort(., decreasing = TRUE) * 100/nrow(wells_features) ->
  missings

missings <- missings[missings != 0]


#' This throws up 7 columns with missing values. The most serious deficiency is
#' within `scheme_name`. It is very difficult to impute columns of datatype
#' string. We drop `scheme_name` and see if we can impute the rest using the
#' most common value. An alternate model is to add in NA as an additional
#' feature

COLS_TO_DROP <- names(missings)[1]

#' We begin by looking at signs of class imbalance among the labels

ggplot(wells_labels) +
  geom_bar(aes(x = factor(status))) +
  scale_x_discrete('Well status')

#' The above plot shows signs of class imbalance within the three categories
#' of wells

names(wells_features)
glimpse(wells_features)

#' We look at the distribution of individual features for skew

ggplot(wells_features) +
  geom_histogram(aes(x = amount_tsh))

#' This shows a highly skewed distribution. Apply a log

ggplot(wells_features) +
  geom_histogram(aes(x = amount_tsh)) +
  scale_x_log10()

#' logarithm smoothens the distribution a lot. But take care that there are a lot
#' of zeros, as noted below

sum(wells_features$amount_tsh == 0)

#' There seems to be a lot of funders for the well. This might be an important
#' feature
 
ggplot(wells_features) +
  geom_bar(aes(x = funder)) +
  coord_flip()

length(unique(wells_features$funder))

#' Close to 1900 unique funders, some of them contribute to disproportionately
#' large number of wells; this is a prime feature for bucketing into 3-4 categories

table(wells_features$funder) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame() %>% 
  mutate(perc_total = n  * 100/nrow(wells_features))

#' We can divide this into Local Government (LG), Heavy Donors (HD), Others (O)

