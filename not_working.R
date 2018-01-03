
# Multivariate Forecasting with ML stacking (NOTE: Not working due to change in creation of ensemble learners)

One difficulty in the integration of machine learning and forecasting is that most machine learning models rely on a large amount of exogenous data to derive good answers. As stated before, since we can't forecast these exogenous variables it becomes difficult to fully use the power of machine learning models. One workaround to this is to use mlr's ensembling method to build a multivariate forecaster, targeting a specific variable, and then training a machine learning model on the forecasts of all variables. It works as the following

1. Make a multivariate task, targeting the chosen variable
2. Create a stacked learner, whose base learner is a `mfcregr.BigVAR` model
3. Create a super learner, which will be trained on the 'stacked' learner
4. Tune and train the stacked learner
5. Tune and train the super learner on the forecasts produced by the stacked learner

And now your final model you use for prediction will first forecast your extra variables and then use those forecasts to forecast the final values for the target variable.

```{r multivarStack, cache = TRUE}
EuStockMarkets.train = EuStockMarkets[1:1855,]
EuStockMarkets.test = EuStockMarkets[1856:1860,]
multfore.task = makeMultiForecastRegrTask(id = "bigvar", data = EuStockMarkets.train, target = "FTSE")

resamp.sub = makeResampleDesc("GrowingCV",
                              horizon = 5L,
                              initial.window = .97,
                              size = nrow(getTaskData(multfore.task)),
                              skip = .01
)

resamp.super = makeResampleDesc("CV", iters = 3)

base = c("mfcregr.BigVAR")
lrns = lapply(base, makeLearner)
lrns = lapply(lrns, setPredictType, "response")
lrns[[1]]$par.vals$verbose = FALSE
stack.forecast = makeStackedLearner(base.learners = lrns,
                                    predict.type = "response",
                                    super.learner = makeLearner("regr.earth", penalty = 2),
                                    method = "growing.cv",
                                    resampling = resamp.sub)

ps = makeParamSet(
  makeDiscreteParam("mfcregr.BigVAR.p", values = 5),
  makeDiscreteParam("mfcregr.BigVAR.struct", values = "Basic"),
  makeNumericVectorParam("mfcregr.BigVAR.gran", len = 2L, lower = 25, upper = 26),
  makeDiscreteParam("mfcregr.BigVAR.h", values = 5),
  makeDiscreteParam("mfcregr.BigVAR.n.ahead", values = 5)
)

## tuning
multfore.tune = tuneParams(stack.forecast, multfore.task, resampling = resamp.sub,
                           par.set = ps, control = makeTuneControlGrid(),
                           measures = multivar.mase, show.info = FALSE)
multfore.tune
stack.forecast.f  = setHyperPars2(stack.forecast,multfore.tune$x)
multfore.train = train(stack.forecast.f,multfore.task)
multfore.train
multfore.pred = predict(multfore.train, newdata = EuStockMarkets.test)
multfore.pred
performance(multfore.pred, mase, task = multfore.task)

```