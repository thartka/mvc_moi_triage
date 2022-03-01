############################################################
#
#  Project: Evaluate MOI criteria for MVCs
#
#  This script assesses for a difference in AUROCs for 
#   2000-2009 vs 2010-2019 using the Delong method.
#
###########################################################

rm(list=ls())

library(tidyverse)
library(pROC)

# load nass and ciss with imputation
mvcs <- read_csv("./Data/NASS_CISS-2000_2019-imputated.csv")

# select the first set of imputated cases since the 
#  analysis of all five imputation sets would falsely 
#  tightness 95 CIs
mvcs <- mvcs[0:150684,]

# set variable for sequential test of CDC criteria
mvcs <- mvcs %>% 
  mutate(cdc_cit = case_when(
    int18 == 1 ~ 1,
    int12occ == 1 ~ 2,
    ejection == 1 ~ 3,
    other_death == 1 ~ 4,
    TRUE ~ 0
  )
)

# separate into decades
mvcs2000 <- mvcs %>% filter(year<2010)
mvcs2010 <- mvcs %>% filter(year>=2010)

# get ROCs
roc2000 <- roc(mvcs2000$iss16, mvcs2000$cdc_cit, ci=TRUE)
roc2010 <- roc(mvcs2010$iss16, mvcs2010$cdc_cit, ci=TRUE)

# examine AUCs
roc2000$auc
roc2010$auc

roc2000$ci
roc2010$ci

# compare AUCs
roc.test(roc2000, roc2010)

# get CI
roc.test(roc2000, roc2010, conf.level=0.95)
