#################################################
#
# Simple preprocessing script for AdultUCI data
#
#################################################

library(arules)
library(dplyr)
library(purrr)
library(reshape2)
library(tidyr)
library(caret)

data("AdultUCI")
AdultUCI <- AdultUCI[!is.na(AdultUCI$income), ]

outcome <- AdultUCI[, "income"]
input <- AdultUCI[, !names(AdultUCI) %in% "income"]

# remove order to factor for education
class(input$education) <- "factor"

# could just delete lines with missing data -
# but choose to change NA's to not knowns (NK)
input[] <- lapply(input[], function(x) {
  if(class(x) == "factor") {
    x <- addNA(x, ifany = TRUE)
    levels(x)[levels(x) %in% NA] <- "NK"
    return(x)
  } else {
    return(x)
  }
}) 

# train-test split
set.seed(1234)
inTrain <- createDataPartition(outcome, p = 0.7, list = FALSE)
outcomeTest <- outcome[-inTrain]
outcomeTrain <- outcome[inTrain]
inputTest <- input[-inTrain, ]
inputTrain <- input[inTrain, ]

# now split training into train/validation
set.seed(1234)
inTrain <- createDataPartition(outcomeTrain, p = 0.9, list = FALSE)
outcomeVal <- outcomeTrain[-inTrain]
outcomeTrain <- outcomeTrain[inTrain]
inputVal <- inputTrain[-inTrain, ]
inputTrain <- inputTrain[inTrain, ]

facCols <- names(inputTrain[, vapply(inputTrain, class, character(1)) == "factor"])
intCols <- names(inputTrain[, vapply(inputTrain, class, character(1)) == "integer"])

inputTrain$set <- "train"
inputVal$set <- "validation"
inputTest$set <- "test"

inputSet <- rbind(inputTrain, inputVal)
inputSet <- rbind(inputSet, inputTest)

#
# try and keep things tidy with some functional programming
byGroup <- inputSet %>%
  group_by(set) %>% 
  nest() 

#
# remove the education-num column input, as contains same info as education
# cast country to united states or other
# transform fnlwgt to resolve skewness (probably not needed for C50, but may improve
# more sensitve models)
byGroup <- byGroup %>%
  mutate(data = map(data, function(x) x[, !names(x) %in% "education-num"]) ) %>%
  mutate(data = map(data,
                    function(x) {
                      x["native-country"] = ifelse(x["native-country"] == "United-States",
                                                          "United-States",
                                                          "Other")
                      x
                    })) %>%
  mutate(data = map(data,
                    function(x) {
                      x["fnlwgt"] = sqrt(x["fnlwgt"])
                      x
                    })) %>%
  unnest()

#
# remove
intCols <- intCols[!intCols == "education-num"]

#
# create dummy variable based on inputTrain
catDummies <- dummyVars(~. ,
                        data = byGroup[byGroup$set == "train", facCols],
                        levelsOnly = FALSE,
                        fullRank = FALSE)

# 
# collect and continue
# make dummy variables
# remove facCols
# remove one of dichotomoous variables
byGroup <- byGroup %>%
  group_by(set) %>%
  nest() %>%
  mutate(data = map(data, function(x) {
    dummies <- data.frame(predict(catDummies, x[, facCols]))
    x <- cbind.data.frame(x, dummies)
    x[, !names(x) %in% facCols]
  })) %>% 
  mutate(data = map(data, function(x) {
    x[, !names(x) %in% c("sex.Female", "X.native.country.Other") ]
  })) %>% 
  unnest()
    
#
# now lets search for non informative and very highly correlated predictors
nzVar <- nearZeroVar(byGroup[byGroup$set == "train", 2:ncol(byGroup)], saveMetric = TRUE)
fullSet <- rownames(nzVar[nzVar$zeroVar == FALSE, ])

trainCorr <- cor((byGroup[byGroup$set == "train", fullSet]))
fullCorr <- findCorrelation(trainCorr[, fullSet], cutoff = 0.99)
fullCorrNames <- names(byGroup)[fullCorr]

#
# occupation.Transport.moving is nearly fully correlated with
# another variable, so remove
fullSet <- fullSet[fullSet != fullCorrNames]

#
# pull out train, validation and test set
inputTrain <- byGroup[byGroup$set == "train", fullSet]
inputVal <- byGroup[byGroup$set == "validation", fullSet]
inputTest <- byGroup[byGroup$set == "test", fullSet]
