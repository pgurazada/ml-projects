#' ---
#' title: "Select relevant features"
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Tue May 22 13:58:57 2018
#' 
#' In this script, we implement the feature selection from exploratory analysis
#' to assemble the training data set 

library(tidyverse)
library(rsample)

set.seed(20130810)

wells_features <- read_csv('wells-africa/data/well_features.csv', 
                           progress = TRUE)

wells_labels <- read_csv('wells-africa/data/well_labels.csv',
                         progress = TRUE)

wells_labels %>% 
  
  mutate(status = case_when(status_group == "functional" ~ 2,
                            status_group == "non functional" ~ 0,
                            TRUE ~ 1)) %>% 
  
  select(status) ->
  
  wells_labels_train

gps_ht_bins <- predict(discretize(wells_features$gps_height), wells_features$gps_height)
constr_yr_bins <- predict(discretize(wells_features$construction_year), wells_features$construction_year)

wells_features %>% 
  
  select(-id, -scheme_name) %>% # too many missing values
  
  mutate(tsh = case_when(amount_tsh == 0 ~ 0,
                         TRUE ~ log(amount_tsh)),
         tsh_zero = case_when(amount_tsh == 0 ~ 1,
                              TRUE ~ 0)) %>% # split the tsh into zeros and log of non-zeros
  
  select(-amount_tsh) %>% # deselect original feature
  
  mutate(funded_by = case_when(funder == 'Government Of Tanzania' ~ 'Govt',
                               funder == 'Danida' ~ 'F1',
                               funder == 'Hesawa' ~ 'F2',
                               funder == 'Rwssp' ~ 'F3',
                               funder == 'World Bank' ~ 'F4',
                               funder == 'Kkkt' ~ 'F5',
                               funder == 'World Vision' ~ 'F6',
                               funder == 'Unicef' ~ 'F7',
                               funder == 'Tasaf' ~ 'F8',
                               funder == 'District Council' ~ 'F9',
                               TRUE ~ 'Oth')) %>% # split values into top 10 and the rest
  
  select(-funder) %>% # deselect original feature
  
  mutate(max_date = max(date_recorded),
         yrs_since_last_collection = as.numeric(max_date - date_recorded)/365) %>% 
  
  mutate(data_collec_at = case_when(yrs_since_last_collection <= 2 ~ 'LT2Y',
                                    yrs_since_last_collection > 2 & yrs_since_last_collection <= 4 ~ 'BTW24Y',
                                    TRUE ~ 'GT4Y')) %>% 
  
  select(-date_recorded, -max_date, -yrs_since_last_collection) %>% # remove original and intermeidate features
  
  mutate(gps_height = gps_ht_bins) %>% # use bins from global scope and overwrite numeric variable
  
  mutate(installer_cat = case_when(installer == 'DWE' ~ 'DWE',
                                   TRUE ~ 'OTH')) %>% # bin into two groups
  
  select(-installer) %>% # remove original feature
  
  mutate(wpt_name_cat = case_when(wpt_name == 'none' ~ 'UN',
                                  TRUE ~ 'N')) %>% # split into two groups
  
  select(-wpt_name) %>% # remove original feature

  mutate(num_private_cat = case_when(num_private == 0 ~ 'Z',
                                     TRUE ~ 'NZ')) %>% # Split into zeros and non-zeros
  
  select(-num_private) %>% # remove original feature
  
  mutate(subvillage_cat = case_when(subvillage %in% c('Madukani', 'Shulengi', 'Majengo') ~ 'T1',
                                    subvillage %in% c('Kati', 'Mtakuja', 'Sokoni') ~ 'T2',
                                    TRUE ~ 'T3')) %>% # split into 3 groups
  
  select(-subvillage) %>% # remove original feature
  
  select(-region_code, -district_code) %>% 
  
  mutate(ward_cat = case_when(ward %in% c('Igosi') ~ 'T1',
                              ward %in% c('Imalinyi', 'Siha Kati', 'Mdandu', 'Nduruma', 'Kitunda', 'Mishamo', 'Msindo') ~ 'T2',
                              TRUE ~ 'T3')) %>% # split into three categories
  
  select(-ward) %>% # remove original feature
  
  mutate(pop = case_when(population == 0 ~ 0,
                         TRUE ~ log(population)),
         pop_zero = case_when(population == 0 ~ 1,
                              TRUE ~ 0)) %>% # split into zero population and log(nonzeropopulation)
  
  select(-population) %>% #drop original feature
  
  select(-recorded_by) %>% # single valued
  
  mutate(construction_year = constr_yr_bins) %>% # overwrite with bins
  
  select(-extraction_type_class) %>% # redundant feature
   
  select(-management) %>% # redundant feature
  
  select(-payment) %>% # redundant feature
  
  select(-water_quality, -quantity) %>% # redundant feature
  
  select(-source) %>% # redundant feature
  
  select(-waterpoint_type_group) ->
  
  wells_features_train

glimpse(wells_features_train)  
glimpse(wells_labels_train)
  
write_csv(x = wells_features_train, 'wells-africa/processed/wells_features_train.csv')
write_csv(x = wells_labels_train, 'wells-africa/processed/wells_labels_train.csv')

  
  
  
