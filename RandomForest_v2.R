#All packages used
library(tidyverse) 
library(randomForest)
library(caret)
library(ROSE)
library(pdp)
library(lime)

#==========================================================
#creating response variable
#==========================================================
new_rec<-mutate(credit_record, Risk = case_when(
  MONTHS_BALANCE<=0 & STATUS=="C" | STATUS=="X" | STATUS=="0"    ~   "No",
  MONTHS_BALANCE<=0 & STATUS=="1" | STATUS=="2" |STATUS=="3" |STATUS=="4" |STATUS=="5" ~ "Yes"))

new_rec<-new_rec%>%
  group_by(ID)%>%
  summarise(Risk= max(Risk))
#==========================================
#Merging two tables by ID using inner join
#==========================================
new_credit<- merge(application_record, new_rec, by="ID")
new_credit$Risk<-as.factor(new_credit$Risk)
new_credit11=new_credit


new_credit11$FLAG_MOBIL<-as.factor(new_credit11$FLAG_MOBIL)
new_credit11$FLAG_WORK_PHONE<-as.factor(new_credit11$FLAG_WORK_PHONE)
new_credit11$FLAG_PHONE<-as.factor(new_credit11$FLAG_PHONE)
new_credit11$FLAG_EMAIL<-as.factor(new_credit11$FLAG_EMAIL)
#===============================================
#Splitting to train and test
#===============================================
set.seed(55)
in_training <- createDataPartition(new_credit11$Risk, p = .70, list = FALSE)
Trainset <- new_credit11[in_training, ]
Testset <- new_credit11[-in_training, ]
#==========================#=====================
#Sampling Method
rover<- ovun.sample(Risk~., data = Trainset, method = "over", 
                    N=45034, seed = 123)$data
runder<- ovun.sample(Risk~., data = Trainset, method = "under",
                     N=6008, seed = 123)$data
rboth<- ovun.sample(Risk~., data = Trainset, method = "both", 
                    p=0.5, seed = 123)$data


#================
#Training
#================
rfmodel <- randomForest(Risk~., data = Trainset, 
                        importance = TRUE)
rfmodel

rfover <- randomForest(Risk~., data = rover, 
                       importance = TRUE)
rfover

rfunder <- randomForest(Risk~., data = runder,
                        importance = TRUE)
rfunder

rfboth <- randomForest(Risk~., data = rboth, 
                       importance = TRUE)
rfboth


#================================
#Tuning random forest
#================================
rfoverT1 <- tuneRF(rfover[,-17], rfover[,17],
                   stepFactor = 0.5,
                   plot = TRUE,
                   ntreeTry = 500,
                   mtry=8,
                   trace = TRUE,
                   improve = 0.05)

#================================
#10-fold cross validation
#================================
fit_control <- trainControl(
  method = "cv",
  number = 10
)
set.seed(45)
rf_Tune <- train(Risk ~ ., 
                 data = rover,
                 method= "rf",
                 tuneGrid=data.frame(mtry=8),
                 trControl = fit_control)
rf_Tune

pred<- predict(rf_Tune, Testset)
confusionMatrix(pred, Testset$Risk)

#=============================
#Variable importance
#=============================
imp<-varImpPlot(rfoverT1, sort =TRUE, col=1, scale = TRUE)

#Retraining random forest by removing three least important variables
retrain<-Trainset%>%
  select(-FLAG_EMAIL, -FLAG_OWN_REALTY, -FLAG_WORK_PHONE,-CODE_GENDER)
rover1<- ovun.sample(Risk~., data = retrain, method = "over", 
                     N=45034, seed = 123)$data

rfmodel2 <- randomForest(Risk~., data = rover1, mtry=8,
                         ntree= 500, importance = TRUE)
rfmodel2
#=====================================
#One-way paritial dependencies plot
#=====================================
pd1<- partial(rfoverT1, pred.var = "AMT_INCOME_TOTAL", which.class = "Yes",
              plot = TRUE, rug = TRUE)
pd2<- partial(rf_balTune, pred.var = "DAYS_EMPLOYED", which.class = "Yes",
              plot = TRUE, rug = TRUE)
pd3<- partial(rf_balTune, pred.var = "DAYS_BIRTH", which.class = "Yes",
              plot = TRUE, rug = TRUE)
grid.arrange(pd1, pd2, pd3, ncol=2)

pd5<- partial(rf_balTune, pred.var = "NAME_HOUSING_TYPE", plot = TRUE, 
              rug = TRUE, which.class = "No")
pd6<- partial(rf_balTune, pred.var = "NAME_FAMILY_STATUS", plot = TRUE,
              rug = TRUE, which.class = "No")
pd7<- partial(rf_balTune, pred.var = "NAME_EDUCATION_TYPE", plot = TRUE,
              rug = TRUE, which.class = "No")
grid.arrange(pd5, pd6, pd7, ncol=1)
#=====================================
#Two-way partial dependency plot
#=====================================

pdd <- partial(rfoverT1, pred.var = c("AMT_INCOME_TOTAL", "NAME_FAMILY_STATUS"), 
               plot = TRUE, which.class = "Yes", type = "classification")

pdd1 <- partial(rfoverT1, pred.var = c("AMT_INCOME_TOTAL", "DAYS_BIRTH"), 
                plot = TRUE, which.class = "Yes", type = "classification",
                probs = TRUE, plot.engine = "ggplot2", rug = TRUE,
)

#==================================
#LIME explanation 
#==================================
y<-Trainset%>%
  select(-Risk)
x<-Testset%>%
  select(-Risk)%>%
  sample_n(size =500)

explainer <- lime(y, rf_Tune)
explanation<- explain(x, explainer, labels = "Yes", n_features = 7)
explanation1<- explain(x, explainer, labels = "No", n_features = 7)

L1<-plot_explanations(explanation1)

L2<-plot_features(explanation, ncol = 1, cases =1:4)
L3<-plot_features(explanation1, ncol = 2, cases =1:4)
