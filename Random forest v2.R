library(tidyverse) 
library(randomForest)
library(lime)
library(gmodels)
library(mosaic)
library(caret)
library(ROSE)
library(ranger)
library(pdp)
library(gridExtra)
library(grid)
library(ggridges)
library(ggthemes)
library(pander)
library(xgboost)
library(yardstick)
#library(Information)

#==========================================================
#creating response varible

new_rec<-mutate(credit_record, Risk = case_when(
  MONTHS_BALANCE<=0 & STATUS=="C" | STATUS=="X" | STATUS=="0"    ~   "No",
  MONTHS_BALANCE<=0 & STATUS=="1" | STATUS=="2" |STATUS=="3" |STATUS=="4" |STATUS=="5" ~ "Yes"))
head(new_rec)
new_rec<-new_rec%>%
  group_by(ID)%>%
  summarise(Risk= max(Risk))

#Merging the two tables by id
new_credit<- merge(application_record, new_rec, by="ID")
new_credit$Risk<-as.factor(new_credit$Risk)
new_credit11=new_credit

#=========================================================
#Renaming columns

#=========================================================
new_credit11$FLAG_MOBIL<-as.factor(new_credit11$FLAG_MOBIL)
new_credit11$FLAG_WORK_PHONE<-as.factor(new_credit11$FLAG_WORK_PHONE)
new_credit11$FLAG_PHONE<-as.factor(new_credit11$FLAG_PHONE)
new_credit11$FLAG_EMAIL<-as.factor(new_credit11$FLAG_EMAIL)

#==========================================================
#Missing data(data which is null, FLAG_MOBIL_PHONE doesn't contrbiute), 
#bad cardinality(high number of rows from two tables), 
#outliers 
#Imputation: replacing null value (measure of centeral tendency. i.e mean)



#==========================================================
#Information value and weight of evidence 
#factor_vars <- c ("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
#                  "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE")  # get all categorical variables

#all_iv <- data.frame(VARS=factor_vars, IV=numeric(length(factor_vars)), STRENGTH=character(length(factor_vars)), stringsAsFactors = F)  # init output dataframe
#for (factor_var in factor_vars){
 # all_iv[all_iv$VARS == factor_var, "IV"] <-IV(X=new_credit11[, factor_var], Y=new_credit11$Risk)
  #all_iv[all_iv$VARS == factor_var, "STRENGTH"] <- attr(IV(X=new_credit11[, factor_var], Y=new_credit11$Risk), "howgood")
#}
#all_iv <- all_iv[order(-all_iv$IV), ]  # sort
#variable reduction via information value and weight of evidence
#FLAG_MOBIL is less 

plot1 <- new_credit11 %>%
  ggplot(aes(x = Risk, fill = Risk)) +
  scale_fill_tableau() +
  geom_bar(alpha = 0.8) 
  #guides(fill = FALSE)
 tbl_df(new_credit11)
 
 new_credit11 %>%
   summarize(mean_cty = mean(CNT_FAM_MEMBERS), 
             st_dev_cty = sd(CNT_FAM_MEMBERS)) %>% 
              pander()
 
 new_credit11<-select(new_credit11, -FLAG_MOBIL,-ID)

#==========================#===============================
#Splitting to train and test

set.seed(55)
in_training <- createDataPartition(new_credit11$Risk, p = .70, list = FALSE)
Trainset <- new_credit11[in_training, ]
Testset <- new_credit11[-in_training, ]

#==========================#========================================
rover<- ovun.sample(Risk~., data = Trainset, method = "over", N=45034, seed = 123)$data
runder<- ovun.sample(Risk~., data = Trainset, method = "under", N=6008, seed = 123)$data
rboth<- ovun.sample(Risk~., data = Trainset, method = "both", p=0.5, seed = 123)$data
#=============================#===================================================================
#Training and Prediction

fit_control <- trainControl(
  method = "cv",
  number = 10
  )
set.seed(45)
rf_balTune <- train(Risk ~ ., 
                    data = rover,
                    method= "rf",
                    tuneGrid=data.frame(mtry=5),
                    trControl = fit_control)
rf_balTune


pred_rf<- predict(rf_balTune, Testset)
conM<-confusionMatrix(pred_rf, Testset$Risk)
fourfoldplot(conM$table, color = c("#34eb89", "#6699CC"), conf.level = .95,
             std = c("margins", "ind.max", "all.max"), margin = c(1,2),
             space = 0.2, main = "Confusion matrix" )

p1<-roc.curve(Testset$Risk, pred_rf)
#=========================================================
rfmodel <- randomForest(Risk~., data = rover,importance = TRUE)
plot(rfmodel$err.rate)

plot(confusionM)

#Error rate
#============================================================
 var_imp<-varImp(rf_balTune, scale = FALSE)
 plot(var_imp, top = 20)
 filterVarImp(Trainset, Trainset$Risk)
#paritial dependencies plot
pd1<- partial(rf_balTune, pred.var = "AMT_INCOME_TOTAL", plot = TRUE,
                                   rug = TRUE)
pd2<- partial(rf_balTune, pred.var = "AMT_INCOME_TOTAL", plot = TRUE,
        plot.engine = "ggplot2")
grid.arrange(pd1, pd2, ncol=2)

pd3<- rf_balTune%>%
      partial(pred.var="AMT_INCOME_TOTAL")%>%
      plotPartial(smooth=TRUE, lwd=2, ylab= expression(f(AMT_INCOME_TOTAL)),
                  main= "AMT_INCOME PDP")
#=====================#==================================
#LIME explanation 
y<-Trainset%>%select(-Risk)
x<-Testset%>%select(-Risk)%>%sample_n(size =100)
explainer <- lime(y, rf_balTune)
# feature selection method tree 
explanation<- explain(x, explainer, labels = "Yes", n_features = 7, 
                      feature_select="trees")
explanation

L1<-plot_explanations(explanation)

L2<-plot_features(explanation, ncol = 2, cases =1:4)

#=========================================================









#save the plot in a variable and that will help you on saving time
# Tuning the parameter 
#Looking at the dataset in kaggle
# Tune mtry, ntry, numbers on the traincontrol, 
# Have more time after hand in 
# Lime implementation mathematically...implement 
#it to the model that you have and don't need more as
#in the package
# PCA on the plote is because of the preprocess on the train 
#Trying all the months instead of only 6 months 
# Look how I create the response variable 
# There are a big number on the confusion matrix because I use 
# a testset which is not sampled ...maybe check it by sampling 
#the testset before having the confusion matrix 
#Finding the best tree in the random forest on the kaggle 
#kernal that the guy he has becausehe has good accuracy.





#Notes from Kaggle dataset
# 1. overdue more than 60 days are "1" else "0"
# 