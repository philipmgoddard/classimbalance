#################################################
#
# Investigate methods to remedy class imbalance using
# AdultUCI dataset. Assume that input and oucome obtained
# using the preprocessing script provided.
#
# Use C50 to demonstrate some techniques
#
#################################################

library(caret)
library(pROC)
library(doMC)
registerDoMC(cores = 4)

#
# define summary functions for model training
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))

fourStats <- function (data, lev = levels(data$obs), model = NULL) {
  accKapp <- postResample(data[, "pred"], data[, "obs"])
  out <- c(accKapp,
           sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out)[3:4] <- c("Sens", "Spec")
  out
}


# positive class is "small"
prop.table(table(outcomeTrain))

# we are proobably more interested in
# the minority class, so lets do a hack to reorder (alphabetical by default)
outcomeTrain <- as.factor(ifelse(outcomeTrain == "small", "small", "large"))
outcomeVal <- as.factor(ifelse(outcomeVal == "small", "small", "large"))
outcomeTest <- as.factor(ifelse(outcomeTest == "small", "small", "large"))

#
# set up train control
ctrl <- trainControl(method = "cv",
                     number = 3,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     summaryFunction = fiveStats)

ctrlNoProb <- ctrl
ctrlNoProb$summaryFunction <- fourStats
ctrlNoProb$classProbs <- FALSE

# grid of tuning params
c50Grid <- expand.grid(trials = c(1:9, (1:10) * 10),
                       model = c("tree", "rules"),
                       winnow = c(TRUE, FALSE))

#################################################
#################################################
#
# baseline 1): tune for ROC AUC
set.seed(476)
c50Tune <- train(x = inputTrain,
                 y = outcomeTrain,
                 method = "C5.0",
                 tuneGrid = c50Grid,
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl)

# sens 0.6722, ,spec 0.9312
confusionMatrix(predict(c50Tune, inputTest), outcomeTest)

#
# baseline 2): tune for sensivity
set.seed(476)
c50TuneBaseline <- train(x = inputTrain,
                         y = outcomeTrain,
                         method = "C5.0",
                         tuneGrid = c50Grid,
                         verbose = FALSE,
                         metric = "Sens",
                         trControl = ctrl)

#
# test set performance
# sens = 0.6509, spec = 0.9347
confusionMatrix(predict(c50TuneBaseline, inputTest), outcomeTest)

#
# ROC curve
C50ROC <- roc(outcomeTest, predict(c50TuneBaseline, inputTest, type = "prob")[[1]],
             levels = rev(levels(outcomeTest)))
plot(C50ROC, legacy.axes = TRUE)

#
# build up a data frame for results
testResults <- data.frame(outcome = outcomeTest,
                         baselinePred =  predict(c50TuneBaseline, inputTest))


#################################################
#################################################

#
# upsample minority class
set.seed(1103)
upSampledTrain <- upSample(x = inputTrain,
                               y = outcomeTrain,
                               yname = "earning")

#
# confirm classes now balanced
prop.table(table(upSampledTrain$earning))

set.seed(476)
c50TuneUS <- train(x = upSampledTrain[, !names(upSampledTrain) %in% "earning"],
                   y = upSampledTrain$earning,
                   method = "C5.0",
                   tuneGrid = c50Grid,
                   verbose = FALSE,
                   metric = "Sens",
                   trControl = ctrl)

#
# test set performance
# sens 0.7483 spec 0.8847
confusionMatrix(predict(c50TuneUS, inputTest), outcomeTest)

#
# ROC curve
C50ROCUS <- roc(outcomeTest, predict(c50TuneUS, inputTest, type = "prob")[[1]],
              levels = rev(levels(outcomeTest)))
plot(C50ROCUS)

testResults$US <- predict(c50TuneUS, inputTest)

#################################################
#################################################

#
# downsample majority class
set.seed(1103)
downSampledTrain <- downSample(x = inputTrain,
                           y = outcomeTrain,
                           yname = "earning")

prop.table(table(upSampledTrain$earning))

#
# be careful- check again for nzv predictors!
nzv <- nearZeroVar(downSampledTrain[, 1:(ncol(downSampledTrain) - 1)],
                   saveMetrics = TRUE)
inputDS <- downSampledTrain[, !nzv$zeroVar]

set.seed(476)
c50TuneDS <- train(x = downSampledTrain[, !names(downSampledTrain) %in% "earning"],
                   y = downSampledTrain$earning,
                   method = "C5.0",
                   tuneGrid = c50Grid,
                   verbose = FALSE,
                   metric = "Sens",
                   trControl = ctrl)

# sens 0.8661, spec = 0.7979
confusionMatrix(predict(c50TuneDS, inputTest), outcomeTest)

# validation set ROC curve
C50ROCDS <- roc(outcomeTest, predict(c50TuneDS, inputTest, type = "prob")[[1]],
              levels = rev(levels(outcomeTest)))
plot(C50ROCDS, legacy.axes = TRUE)

testResults$DS <- predict(c50TuneDS, inputTest)

#################################################
#################################################

#
# alternate class cutoff threshold according to ROC
# use validation set to deduce cutoff or risk overfitting
# note validation set also useful if we were looking to calibrate probabilities etc

valPred <- predict(c50TuneBaseline, inputVal, type = "prob")[[1]]
valROC <- roc(outcomeVal, valPred, levels = rev(levels(outcomeTest)))

C50Thresh <- coords(valROC, x = "best", best.method = "closest.topleft")

testResults$alt_cutoff <- factor(ifelse (
  predict(c50TuneBaseline,
          inputTest,
          type = "prob")[[1]] > C50Thresh[1],
  "large",
  "small") )

# test set results
# sens = 0.8469, spec = 0.8265
confusionMatrix(testResults$alt_cutoff, testResults$outcome)


#################################################
#################################################

#
# weights in model tuning. This choice penalises
# false negatives 10 times as much as false positives.
# Note that choices of weights are themselves a tuning param!

c5Matrix <- matrix(c(0, 10, 1, 0), ncol = 2)
rownames(c5Matrix) <- levels(outcomeTrain)
colnames(c5Matrix) <- levels(outcomeTrain)

# optimise kappa as get very poor spec if optimise sens
# note we cannot get probabilities out of C50 if use weights,
# so use different train control object
C5Cost <- train(x = inputTrain,
                y = outcomeTrain,
                method = "C5.0",
                metric = "Kappa",
                cost = c5Matrix,
                tuneGrid = c50Grid,
                verbose = FALSE,
                trControl = ctrlNoProb)

#
# test set results
# sens 0.8372, spec 0.8296
confusionMatrix(predict(C5Cost, inputTest), outcomeTest)

testResults$cost <- predict(C5Cost, inputTest)

# note- cannot get roc curve as probabilities not produced
# when using weights.


#################################################
#################################################

#
# summary plots

toPlot <- data.frame(sensitivity = unlist(lapply(testResults[, 2:6], caret::sensitivity, testResults[, 1])),
                     specificity = unlist(lapply(testResults[, 2:6], caret::specificity, testResults[, 1])))

#toPlot$model <- row.names(toPlot)
toPlot$model <- c("a) baseline", "b) alt. cutoff", "c) upSamp", "d) downSamp", "e) cost")

melted <- reshape2::melt(toPlot, id.vars = "model")

# dot plot for sens/ spec / kappa for each approach
# something like this
ggplot(melted, aes(x = value, y = variable, color = variable)) +
  geom_point(size = 3) +
  facet_grid( model~.) +
  theme_bw() +
  scale_color_manual(values = philTheme()) +
  theme(legend.position = "none") +
  ylab("")

# roc curves for those where can get probabilities out


# todo: roc curve for baseline, up and down


save(c50TuneBaseline, file = "~/Desktop/c50base.Rdata")
save(c50TuneDS, file = "~/Desktop/c50DS.Rdata")
save(C5Cost, file = "~/Desktop/c50Cost.Rdata")
save(c50TuneUS, file = "~/Desktop/c50US.Rdata")
