library(tidyverse) 
library(randomForest)
library(lime)
library(gmodels)
library(mosaic)
library(caret)
library(ROSE)
library(pdp)
library(gridExtra)
library(grid)
library(ggridges)
library(ggthemes)

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
new_credit11$FLAG_MOBIL<-as.factor(new_credit11$FLAG_MOBIL)
new_credit11$FLAG_WORK_PHONE<-as.factor(new_credit11$FLAG_WORK_PHONE)
new_credit11$FLAG_PHONE<-as.factor(new_credit11$FLAG_PHONE)
new_credit11$FLAG_EMAIL<-as.factor(new_credit11$FLAG_EMAIL)
#==========================================================
#Shows class imbalance
plot1 <- new_credit11 %>%
  ggplot(aes(x = Risk, fill = Risk)) +
  scale_fill_tableau() +
  geom_bar(alpha = 0.8) 
#guides(fill = FALSE)
tbl_df(new_credit11)
#=========================================================
new_credit11<-select(new_credit11, -FLAG_MOBIL,-ID)
#==========================#===============================
#Splitting to train and test
set.seed(55)
in_training <- createDataPartition(new_credit11$Risk, p = .70, list = FALSE)
Trainset <- new_credit11[in_training, ]
Testset <- new_credit11[-in_training, ]
#==========================#==================================================
#Sampling Method
rover<- ovun.sample(Risk~., data = Trainset, method = "over", N=45034, seed = 123)$data
runder<- ovun.sample(Risk~., data = Trainset, method = "under", N=6008, seed = 123)$data
rboth<- ovun.sample(Risk~., data = Trainset, method = "both", p=0.5, seed = 123)$data
#=============================#===================================================================
#Training, Prediction and Evaluation

fit_control <- trainControl(
  method = "cv",
  number = 10,
  search = "random"
)

gridd <- expand.grid(mtry = seq(5,8,14) )

set.seed(45)
rf_balTune <- train(Risk ~ ., 
                    data = rover,
                    method= "rf",
                    tuneGrid=gridd,
                    trControl = fit_control)
rf_balTune
#==================================================
#Tunning random forest by changing mtry=3,5,10,15
rf_Tune <- train(Risk ~ ., 
                 data = rover,
                 method= "rf",
                 tuneGrid=data.frame(mtry=10),
                 trControl = fit_control)
rf_Tune
rf_Tune1 <- train(Risk ~ ., 
                  data = rover,
                  method= "rf",
                  tuneGrid=data.frame(mtry=15),
                  trControl = fit_control)
rf_Tune1

rf_Tune2 <- train(Risk ~ ., 
                  data = rover,
                  method= "rf",
                  tuneGrid=data.frame(mtry=3),
                  trControl = fit_control)
rf_Tune2
rf_Tune3 <- train(Risk ~ ., 
                  data = rover,
                  method= "rf",
                  trControl = fit_control)
rf_Tune3
rf_Tune4 <- train(Risk ~ ., 
                  data = runder,
                  method= "rf",
                  trControl = fit_control)
rf_Tune4

pred_rf<- predict(rf_balTune, Testset)
confusionMatrix(pred_rf, Testset$Risk)
p1<-roc.curve(Testset$Risk, pred_rf)
#=========================================================
rf_balunder <- train(Risk ~ ., 
                     data = runder,
                     method= "rf",
                     tuneGrid=data.frame(mtry=5),
                     trControl = fit_control)
rf_balunder

pred_under<- predict(rf_balunder, Testset)
confusionMatrix(pred_under, Testset$Risk)
p3<-roc.curve(Testset$Risk, pred_under)
#===========================================================
rf_both <- train(Risk ~ ., 
                 data = rboth,
                 method= "rf",
                 tuneGrid=data.frame(mtry=5),
                 trControl = fit_control)
rf_both

pred_both<- predict(rf_both, Testset)
confusionMatrix(pred_both, Testset$Risk)
p4<-roc.curve(Testset$Risk, pred_both)
#===========================================================
rfmodel <- randomForest(Risk~., data = rover, mtry=4, ntree=400,
                        importance = TRUE)
rfmodel
pred_f<- predict(rfmodel, Testset)
confusionMatrix(pred_f, Testset$Risk)
p2<-roc.curve(Testset$Risk, pred_f)

#Retrain random forest
xx<-Trainset%>%select(OCCUPATION_TYPE,AMT_INCOME_TOTAL, DAYS_BIRTH,DAYS_EMPLOYED,
                      NAME_INCOME_TYPE,NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS,
                      NAME_HOUSING_TYPE,Risk)
yy<-Testset%>%select(OCCUPATION_TYPE,AMT_INCOME_TOTAL, DAYS_BIRTH,DAYS_EMPLOYED,
                     NAME_INCOME_TYPE,NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS,
                     NAME_HOUSING_TYPE,Risk)
rover1<- ovun.sample(Risk~., data = xx, method = "over", N=45034, seed = 123)$data

rfmodel2 <- randomForest(Risk~., data = rover1,importance = TRUE)
rfmodel2
pred_f1<- predict(rfmodel2, yy)
confusionMatrix(pred_f1, yy$Risk)
#===========================================================
#Trainset[,-17]- all rows and all columns without the response V
# Trainset[,17]- all rows and the response variable
t <- tuneRF(Trainset[,-17], Trainset[,17],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 400,
            trace = TRUE,
            improve = 0.05)
#============================================================
imp<-varImp(rf_balTune, useModel = TRUE, nonpara = TRUE)
plot(imp, top = 15)
filterVarImp(Trainset, Trainset$Risk)
#==========================================================
#paritial dependencies plot
pd1<- partial(rf_balTune, pred.var = "AMT_INCOME_TOTAL", which.class = "Yes",plot = TRUE, rug = TRUE)
pd2<- partial(rf_balTune, pred.var = "DAYS_EMPLOYED", which.class = "Yes",plot = TRUE, rug = TRUE)
pd3<- partial(rf_balTune, pred.var = "DAYS_BIRTH", which.class = "Yes",plot = TRUE, rug = TRUE)
grid.arrange(pd1, pd2, pd3, ncol=2)

#pd4<- partial(rf_balTune, pred.var = "OCCUPATION_TYPE", plot = TRUE, rug = TRUE, which.class = "Yes")
pd5<- partial(rf_balTune, pred.var = "NAME_HOUSING_TYPE", plot = TRUE, rug = TRUE, which.class = "No")
pd6<- partial(rf_balTune, pred.var = "NAME_FAMILY_STATUS", plot = TRUE, rug = TRUE, which.class = "No")
pd7<- partial(rf_balTune, pred.var = "NAME_EDUCATION_TYPE", plot = TRUE, rug = TRUE, which.class = "No")

grid.arrange(pd5, pd6, pd7, ncol=1)


pd <- partial(rfmodel, pred.var = c("AMT_INCOME_TOTAL", "CODE_GENDER"), 
              plot = TRUE, which.class = "Yes", type = "classification")

pdd <- partial(rfmodel, pred.var = c("AMT_INCOME_TOTAL", "NAME_HOUSING_TYPE"), 
               plot = TRUE, which.class = "Yes", type = "classification")

pdd1 <- partial(rfmodel, pred.var = c("AMT_INCOME_TOTAL", "NAME_FAMILY_STATUS"), 
                plot = TRUE, which.class = "Yes", type = "classification")

pdd2 <- partial(rfmodel, pred.var = c("AMT_INCOME_TOTAL", "OCCUPATION_TYPE"), 
                plot = TRUE, which.class = "Yes", type = "classification")
#=====================#========================================
#LIME explanation 
y<-Trainset%>%select(-Risk)
x<-Testset%>%select(-Risk)%>%sample_n(size =500)

explainer <- lime(y, rf_balTune)
explanation<- explain(x, explainer, labels = "Yes", n_features = 5, feature_select = "highest_weight")
explanation1<- explain(x, explainer, labels = "Yes", n_features = 5, feature_select = "tree")

explanation

L1<-plot_explanations(explanation1)

L2<-plot_features(explanation, ncol = 2, cases =79)
L3<-plot_features(explanation1, ncol = 2, cases =79)

#===============================================================