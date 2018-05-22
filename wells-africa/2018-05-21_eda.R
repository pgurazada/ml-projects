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
#' 
#' *1. `amount_tsh`*

ggplot(wells_features) +
  geom_histogram(aes(x = amount_tsh))

#' This shows a highly skewed distribution. Apply a log

ggplot(wells_features) +
  geom_histogram(aes(x = amount_tsh)) +
  scale_x_log10()

#' logarithm smoothens the distribution a lot. But take care that there are a lot
#' of zeros, as noted below

sum(wells_features$amount_tsh == 0)

#' *2. `funder`*
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

#' We can divide this into Local Government (LG), D1 - D_i for donors upto 2% of
#' total and Others (O)
#' 
#' *3. `date_recorded`*
#' 
#' date recorded seems to be not very relevant, but we cannot be sure, we can
#' subtract this from the current date so we have a sense of the distribution of
#' data collection times

wells_features %>% 
  mutate(max_date = max(date_recorded),
         days_since_last_collection = as.numeric(max_date - date_recorded)) %>% 
  select(date_recorded, days_since_last_collection) %>% 
  
  ggplot() +
   geom_bar(aes(x = date_recorded))
  
#' The dates seem to be cluttered in three bands. We can discretize this into 4
#' bins
#' 
#' *4. `gps_height`*
#' Moving on to `gps_height`. This is the altitude of the well. Do higher level
#' wells have more chance of not being functional?

ggplot(wells_features) +
  geom_histogram(aes(x = gps_height))

#' The plot shows a significant skew. Some values seems to be negative. How can 
#' gps height be negative?

sum(wells_features$gps_height < 0)

wells_features %>% 
  filter(gps_height < 0) %>% 
  summarize(mean_ht = mean(gps_height),
            min_ht = min(gps_height),
            max_ht = max(gps_height))

wells_features %>% 
  filter(gps_height > 0) %>% 
  summarize(mean_ht = mean(gps_height),
            min_ht = min(gps_height),
            max_ht = max(gps_height))

#' The negative values are not heavily skewed to the negative (compared to the
#' positives). We might need to discretize this as well so that the distribution
#' is not as badly skewed

ggplot(wells_features) +
  geom_histogram(aes(x = gps_height), bins = 5)

#' 5 bins seem okay for a start; we can use the discretize function from the 
#' `recipes` package to handle all the discretization

# predict(discretize(wells_features$gps_height), wells_features$gps_height)

#' *5. `installer`*
ggplot(wells_features) +
  geom_bar(aes(x = installer)) + 
  coord_flip()

length(unique(wells_features$installer))

table(wells_features$installer) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame() %>% 
  mutate(perc_total = n  * 100/nrow(wells_features))

#' Split into two categories might be a good idea for installer
#' 
#' *6. `longitude` and `latitude`*

ggplot(wells_features) +
  geom_histogram(aes(x = longitude))

sum(wells_features$longitude <= 0)

ggplot(wells_features) +
  geom_histogram(aes(x = latitude))

#' There seem to be a few values clustered around 0 longitude but no major
#' alarms
#' 
#' *7. `wpt_name`*

table(wells_features$wpt_name) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame() %>% 
  mutate(perc_total = n  * 100/nrow(wells_features))

#' Most are none. We can divide this into two categories - named (N) and
#' unnamed(UN)
#' 
#' *8. `num_private`
#' No clue what this feature is about

ggplot(wells_features) +
  geom_histogram(aes(x = num_private), bins = 3)

sum(wells_features$num_private == 0)

#' Looks like two bins should be sufficient - zeros and non-zeros

# predict(discretize(wells_features$num_private), wells_features$num_private)

#' *9. `basin`, `subvillage`, `region`, `region_code`, `district_code`, `lga`, `ward`*
#' 
#' These features are related to the geographic location of the well.

length(unique(wells_features$basin))

ggplot(wells_features) + 
  geom_bar(aes(x = basin)) +
  coord_flip()

#' basin is okay for straight dummification

length(unique(wells_features$subvillage))

table(wells_features$subvillage) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame() %>% 
  mutate(perc_total = n  * 100/nrow(wells_features))

#' There are too many - guess 4-5 bins should be okay for this (split into tiers
#' based on frequency)

length(unique(wells_features$region))

ggplot(wells_features) + 
  geom_bar(aes(x = region)) +
  coord_flip()

#' `region` is straight forward dummification

length(unique(wells_features$region_code))

ggplot(wells_features) + 
  geom_histogram(aes(x = region_code))

length(unique(wells_features$district_code))

ggplot(wells_features) + 
  geom_histogram(aes(x = district_code))

#' region, region_code and district code seem to be correlated?

length(unique(wells_features$lga))

ggplot(wells_features) +
  geom_bar(aes(x = lga)) +
  coord_flip()

#' lga is a straight dummification

length(unique(wells_features$ward))

table(wells_features$ward) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame() %>% 
  mutate(perc_total = n  * 100/nrow(wells_features))

ggplot(wells_features) +
  geom_bar(aes(x = ward)) +
  coord_flip()

#' Once again we can go ahead with a dummification of these 2092 values. Based
#' on this prelliminary analysis, we can shortlist basin, subvillage, region,
#' lga and ward. This will add an insane number of features in the form of
#' dummies. I am assuming that the region_code and district_code are correlated
#' with these features so not include them in this analysis. Depending on the
#' convergence issues we might face with this data, we will see if we can prune
#' these features
#' 
#' *10. `population`*
#' 
#' This is the population around the well

ggplot(wells_features) +
  geom_histogram(aes(x = population)) +
  scale_x_log10()

sum(wells_features$population == 0) 

#' Large number of wells seem to have zero population around them. Break this as a
#' separate feature - zero_pop and log(non_zero_pop)

#' *11. `public_meeting`*
#' No clue what this feature means

ggplot(wells_features) +
  geom_bar(aes(x = public_meeting))

#' make it dummy, take care of the NAs
#' 
#' *12. `recorded_by` *
#' 
#' Who entered this data

length(unique(wells_features$recorded_by))

#' Drop this feature
#' 
#' *13. `scheme_management` *
#' 
#' Who operates this water point

length(unique(wells_features$scheme_management))

ggplot(wells_features) +
  geom_bar(aes(x = scheme_management)) +
  coord_flip()

#' Take care of NA's and dummify
#' 
#' *14. `scheme_name` *

length(unique(wells_features$scheme_name))

table(wells_features$scheme_name) %>% 
  sort(., decreasing = TRUE) %>% 
  as_data_frame()

#' This is a badly captured feature, drop it
#' 
#' *15. `permit` *
#' 
#' If the water point is permitted

length(unique(wells_features$permit))

ggplot(wells_features) +
  geom_bar(aes(x = permit))

#' Take care of NA and dummify
#' 
#' *16. `construction_year` *
#' 
#' When was the well constructed?

length(unique(wells_features$construction_year)) 

ggplot(wells_features %>% filter(construction_year != 0)) +
  geom_bar(aes(x = construction_year))

sum(wells_features$construction_year == 0) 

#' The NA's seem to have been coded as zeros. The rest need to be binned

# predict(discretize(wells_features$construction_year), wells_features$construction_year) 

#' *17. `extraction_type` , `extraction_type_group`, `extraction_type_class` *
#' 
#' Kind of extraction used by the well

length(unique(wells_features$extraction_type))

ggplot(wells_features) +
  geom_bar(aes(x = extraction_type)) +
  coord_flip()

length(unique(wells_features$extraction_type_group))

ggplot(wells_features) +
  geom_bar(aes(x = extraction_type_group)) +
  coord_flip()

length(unique(wells_features$extraction_type_class))

ggplot(wells_features) +
  geom_bar(aes(x = extraction_type_class)) +
  coord_flip()

#' The choice is between `extraction_type` and `extraction_type_class`. leaning
#' towards `extraction_type`
#' 
#' *18. `management`  and `management_group`  *

length(unique(wells_features$management))

ggplot(wells_features) +
  geom_bar(aes(x = management)) +
  coord_flip()

length(unique(wells_features$management_group))

ggplot(wells_features) +
  geom_bar(aes(x = management_group)) +
  coord_flip()

#' drop management and consider management_group
#' 
#' *19. `payment` and `payment_type` * 

length(unique(wells_features$payment))
length(unique(wells_features$payment_type))

ggplot(wells_features) +
  geom_bar(aes(x = payment)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = payment_type)) +
  coord_flip()

#' drop payment, go with payment_type
#' 
#' * 20. `water_quality`, `quality_group`, `quantity`, `quantity_group`  *
#' 

length(unique(wells_features$water_quality))
length(unique(wells_features$quality_group))
length(unique(wells_features$quantity))
length(unique(wells_features$quantity_group))

ggplot(wells_features) +
  geom_bar(aes(x = water_quality)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = quality_group)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = quantity)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = quantity_group)) +
  coord_flip()

#' drop water_quality and quantity, dummify the other two
#' 
#' *21. `source`, `source_type`, `source_class` *

length(unique(wells_features$source))
length(unique(wells_features$source_type))
length(unique(wells_features$source_class))

ggplot(wells_features) +
  geom_bar(aes(x = source)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = source_type)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = source_class)) +
  coord_flip()

#' Keep `source_type` and `source_class`, drop `source`
#' 
#' * 22. `waterpoint_type` and `waterpoint_type_group`  *

length(unique(wells_features$waterpoint_type))
length(unique(wells_features$waterpoint_type_group))

ggplot(wells_features) +
  geom_bar(aes(x = waterpoint_type)) +
  coord_flip()

ggplot(wells_features) +
  geom_bar(aes(x = waterpoint_type_group)) +
  coord_flip()

#' drop `water_point_type_group`, keep `waterpoint_type`

