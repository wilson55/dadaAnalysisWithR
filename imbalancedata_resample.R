#分類不平衡的處理:
#清除記憶體
rm(list = ls())
gc()

library('grid') #DMwR 會用到的套件
library("lattice") #caret,DMwR 會用到的套件
library(ggplot2) #caret會用到的套件

library(dplyr)

library(caret)
library(smotefamily)
library(purrr) #for functional programming (map)
library(pROC) #for AUC calculations
library(ROSE)


library("readxl")       #載入excel檔


setwd("~/Desktop/7研究題目相關/R_演算法實作/資料合併")
trainSet_2010_2021<-read_excel("trainSet_2010_2021.xlsx")
colnames(trainSet_2010_2021) <-c("ym","ar","ac","rt","jo","te","c4","c5","pv","ch5c",
                       "sdtc","bo","bar","sac4","sac5","sac6","sdtac","sre4","sre5","sre6","sdtrt",
                       "lac","lar","esa","esr","epa","epr","mda","mdr","fea","fer","ina","inr",
                       "bea","ber","sea","ser","hda","hdr","aua","aur","moa","mor","via","vir")
#去除不必要欄位:data.frame,#在執行down sample過程中發現 fe_rtn 沒有變異
trainSet_2010_2021 = subset(trainSet_2010_2021, select = -c(ym,ar,c4,ch5c,sac6,sdtac,sre6,sdtrt,fer))

#產生欄位變數
virableList <- colnames(trainSet_2010_2021)
trainSet_2010_2021_data=data.frame(trainSet_2010_2021[,virableList])
#因子化
trainSet_2010_2021_data$c5 <- ifelse(trainSet_2010_2021_data$c5 <= 0, 'false', 'success');


#檢查數據集的類分佈
c5.tab <-table(trainSet_2010_2021_data$c5)

#檢看資料分類的資料佔比
#若出attempt to make a table with >= 2^31 elements，表示資料表的欄位數過多
#prop.table(table(trainSet_2010_2021_data$c5))
#       0        1 
#0.877247 0.122753 
#par(mfrow = c(1, 2)) #設置在同一頁上有2X2圖
#barplot(c5.tab)

########################
# 基本長條圖:報告用
#c5.dataframe <- as.data.frame(c5.tab)
#colnames(c5.dataframe) <- c('class','recodes')
#ggplot(data = c5.dataframe, aes(x = class, y = recodes)) +
#  geom_bar(stat = "identity",width = 0.5) +
#  geom_text(aes(label = recodes), vjust = 1.6, size = 5.5, color = "white")+
#  theme(axis.text=element_text(size=20,face = "bold"),axis.title.x=element_text(size=24),axis.title.y=element_text(size=24))
   # axis.text:改變坐標軸刻度值的字體大小
   # axis.title.x和axis.title.y改變x軸和y軸標題字體大小
########################
#將將資料分為訓練與測試組
#取得總筆數
n <- nrow(trainSet_2010_2021_data) #1947
#設定隨機數種子
seed <-1117
set.seed(seed)
#將數據順序重新排列
trainSet_2010_2021_data <- trainSet_2010_2021_data[sample(n),]
# 建立抽樣樣本
#取出樣本數的idx
t_idx <- sample(seq_len(n), size = round(0.7 * n))

#訓練資料與測試資料比例: 70%建模，30%驗證
traindata <- trainSet_2010_2021_data[t_idx,];nrow(traindata) #1363
testdata <- trainSet_2010_2021_data[ - t_idx,];nrow(testdata) #584

traindata$c5 <- as.factor(traindata$c5)
testdata$c5  <- as.factor(testdata$c5)
#訓練集資料分佈
table(traindata$c5)
#false success 
#1193     170 
prop.table(table(traindata$c5))
#false   success 
#0.8752751 0.1247249 
#測試集資料分佈
table(testdata$c5)
#false success 
#515      69 
prop.table(table(testdata$c5))
#false   success 
#0.8818493 0.1181507
#資料整理結束
###############################################
#計算AUC，因資料不平衡,AUC,pROC
test_roc <-function(model, data){
  roc(data$c5,
      predict(model,data,type='prob')[,'success'])
}
###############################################
#使用 重復5次 10 折交叉分析，來取得超參數
ctrl <-trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE)
###############################################
#建模:訓練集
orig_fit <-train(c5~.,
                 data = traindata,
                 method = 'rf',
                 verbose = FALSE, #是否輸出log, 會比較慢
                 metric = 'ROC',
                 trControl = ctrl)
saveRDS(orig_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/orig_rf_fit2.rda")
#orig_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/orig_rf_fit2.rda")

#取得重要特徵值,可加入權重 weights = c(0.5, 0.5)
#varImp是用基尼指数
(rfImp_orig <-varImp(orig_fit,scale = F))#importance() from randomForest
summary(rfImp_orig$importance)
#Min.    1st Qu. Median   Mean    3rd Qu.  Max
#0.2369  3.6154  7.0888   8.2589  11.5685  19.0543 
plot(rfImp_orig, top = 20);rfImp_orig

#獲得每個預測變量的 ROC 曲線下面積
(roc_imp <- filterVarImp(x = traindata[,c(1:4,6:36)],traindata[,-c(1:4,6:36)], y = traindata$c5))
head(roc_imp)
                  
#使用ROSE的量測函數:precision,recall,F1
pred.rf_orig<-predict(orig_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_orig[,'success'])
#precision: 0.794
#recall: 0.725
#F: 0.379
roc.curve(testdata$c5, pred.rf_orig[,'success'], plotit = F)
#AUC值等于0.980是个很不錯的结果。

#查看F1 及混淆矩陣結果(對驗証集預測)
predictions_test <-predict(orig_fit,newdata = testdata)
confusionMatrix(predictions_test,testdata$c5,mode = "everything",positive = "success")
#             Reference
#Prediction false success
#false     502      19
#success     13      50
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall        Precision  F1       Balanced Accuracy
#0.9452   0.7268   0.72464      0.79365    0.75758  0.84970
########################################################################
#Build weighted model
#在 library("caret")
#這裡應用的GBM模型自身有一個參數weights可以用於設置樣品的權重；
#給每一個觀察值一個權重
class1_weight = (1/table(traindata$c5)[['false']]) * 0.5;class1_weight #0.0004191115
class2_weight = (1/table(traindata$c5)[["success"]]) *0.5;class2_weight #0.002941176
model.weights <- ifelse(traindata$c5 == "false",
                        class1_weight,class2_weight)
#use the same seed to ensure same cross-validation splits
#使用相同的資料row 值
ctrl$seeds <-orig_fit$control$seeds
#樣品少的類分類錯誤給予更高的罰分 (impose a heavier cost when errors are made in the minority class)
weighted_fit <-train(c5~.,
                     data = traindata,
                     method = "rf",
                     verbose = FALSE,
                     weights = model.weights,
                     metric = "ROC",
                     trControl = ctrl)
saveRDS(weighted_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/weighted_rf_fit2.rda")
#weighted_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/weighted_rf_fit2.rda")

(rfImp_weighted <-varImp(weighted_fit,scale = F))#importance() from randomForest
summary(rfImp_weighted$importance)
#Min.    1st Qu. Median   Mean    3rd Qu.  Max
#0.1465  2.2792  3.9854   8.5140  7.1725   67.7259  
plot(rfImp_weighted, top = 20)

#使用ROSE套件的量測函數:precision,recall,F1
pred.rf_weighted<-predict(weighted_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_weighted[,'success'])
#precision: 0.818
#recall: 0.783
#F: 0.400
roc.curve(testdata$c5, pred.rf_weighted[,'success'], plotit = F)
#Area under the curve (AUC): 0.978
predictions_train<-predict(weighted_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     503      15
#success    12      54
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9538   0.7739  0.78261      0.81818    0.80000  0.87965

########################################################################
#Build down-sampled mode
#下採樣:大眾類中剔除一些樣本，或者說只從大眾類中選取部分樣本
set.seed(seed)
down_train <- downSample(x = traindata[, -ncol(traindata)],
                         y = traindata$c5)
colnames(down_train)[36] <- "c5"
down_train$c5 <-as.factor(down_train$c5)
table(down_train$c5)  

down_fit <-train(c5~.,
                 data = down_train,
                 method = "rf",
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl)

saveRDS(down_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/down_rf_fit2.rda")
#down_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/down_rf_fit2.rda")
(rfImp_down_fit <-varImp(down_fit,scale = F))
summary(rfImp_down_fit$importance)
#Min.      1st Qu.  Median    Mean     3rd Qu.    Max
#0.07697  1.48049   4.34393   4.88183  8.10670    11.37799   
plot(rfImp_down_fit, top = 20)

pred.rf_down<-predict(down_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_down[,'success'])
#precision: 0.593
#recall: 0.928
#F: 0.363
roc.curve(testdata$c5, pred.rf_down[,'success'], plotit = F)
#Area under the curve (AUC): 0.975
predictions_train<-predict(down_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     471      5
#success    44      64
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9161   0.6765  0.9275      0.5926    0.7232     0.9210

########################################################################
#Build up-sampled mode
#上採樣:上採樣是把小眾類複製多份
set.seed(seed)
up_train <- upSample(x = traindata[, -ncol(traindata)],
                     traindata$c5)                         
colnames(up_train)[36] <- "c5"
up_train$c5 <-as.factor(up_train$c5)
table(up_train$c5)  
#ctrl$sampling <- 'up'

up_fit <-train(c5~.,
               data = up_train,
               method = "rf",
               verbose = FALSE,
               metric = "ROC",
               trControl = ctrl)
saveRDS(up_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/up_rf_fit2.rda")
#up_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/up_rf_fit2.rda")
up_fit

(rfImp_up <-varImp(up_fit,scale = F))
summary(rfImp_up$importance)
#Min.      1st Qu.  Median    Mean     3rd Qu.    Max
#0.4445  4.1969    8.3279   35.0738  20.7086   279.0808  
plot(rfImp_up, top = 20)

pred.rf_up<-predict(up_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_up[,'success'])
#precision: 0.757
#recall: 0.768
#F: 0.381

roc.curve(testdata$c5, pred.rf_up[,'success'], plotit = F)
#Area under the curve (AUC): 0.975
predictions_train<-predict(up_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     498      16
#success    17      53
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9435   0.7305  0.76812      0.75714    0.76259     0.86755
########################################################################
#Build SMOTE(Synthetic minority sampling technique) mode
#利用已有樣本生成更多樣本
set.seed(seed)
smote_train <- SMOTE(traindata[,c(1:4,6:36)],traindata[,-c(1:4,6:36)])$data #k=5
colnames(smote_train)[36] <- "c5"
smote_train$c5 <-as.factor(smote_train$c5)
table(smote_train$c5) 

smote_fit <-train(c5~.,
                  data = smote_train,
                  method = "rf",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)
saveRDS(smote_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/smote_rf_fit2.rda")
#smote_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/smote_rf_fit2.rda")
smote_fit

(rfImp_smote <-varImp(smote_fit,scale = F))
summary(rfImp_smote$importance)
#Min.      1st Qu.  Median    Mean     3rd Qu.    Max
#0.3963  5.5553    9.0270   34.0279  24.6336   268.9962  
plot(rfImp_smote, top = 20)

pred.rf_smote<-predict(smote_fit,testdata,type='prob')
#評估分類器學習不平衡準確性的指標,threshold = 0.5
accuracy.meas(testdata$c5, pred.rf_smote[,'success'])
#precision: 0.743,預測的精確度
#recall: 0.797
#F: 0.385

roc.curve(testdata$c5, pred.rf_smote[,'success'], plotit = F)
#Area under the curve (AUC): 0.972
predictions_train<-predict(smote_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     496      14
#success    19      55
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9435   0.7371  0.7971      0.74324    0.76923     0.88010

########################################################################
#Build ROSE(Random Over-Sampling Examples)
#使用平滑引導從少數類周圍的特徵空間鄰域中提取人工樣本。
set.seed(seed)
rose_train <- ROSE(c5 ~ ., data  = traindata, seed = seed)$data 
colnames(rose_train)[36] <- "c5"
rose_train$c5 <-as.factor(rose_train$c5)
table(rose_train$c5) 
rose_fit <- train(c5~., 
                      data = rose_train, 
                      method = "rf",
                      verbose = FALSE,
                      metric = "ROC",
                      trControl = ctrl)
saveRDS(rose_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/rose_rf_fit2.rda")
#rose_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/rose_rf_fit2.rda")
rose_fit

(rfImp_rose <-varImp(rose_fit,scale = F))
summary(rfImp_rose$importance)
#Min.      1st Qu.  Median    Mean     3rd Qu.    Max
#2.267  9.953    18.106   20.027       29.428   47.875  
plot(rfImp_rose, top = 20)

pred.rf_rose<-predict(rose_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_rose[,'success'])
#precision: 0.731
#recall: 0.826
#F: 0.388

roc.curve(testdata$c5, pred.rf_rose[,'success'], plotit = F)
predictions_train<-predict(rose_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#Area under the curve (AUC): 0.972
#            Reference
#Prediction false success
#false     494      12
#success    21      57
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9435   0.7433   0.8261      0.7308    0.7755     0.8927
#依據AUC結果，rose-sample採樣所產生的資料集微降

########################################################################
#使用Borderline-SMOTE
#使用smotefamily包 BLSMOTE 邊界SMOTE算法(Borderline-SMOTE)生成合成正實例
#K=5,C=5 使用 type1
#K : 採樣過程中最近鄰的數目
#C : 計算安全水平過程中的最近鄰數
#訓練集
set.seed(seed)
train_blsmote = BLSMOTE(traindata[,c(1:4,6:36)],traindata[,-c(1:4,6:36)])$data #k=5,c=5
colnames(train_blsmote)[36] <- "c5"
train_blsmote$c5 <-as.factor(train_blsmote$c5)
prop.table((table(train_blsmote$c5)))
#   false   success 
#0.5016821 0.4983179 

blsmote_fit <-train(c5~.,
                    data = train_blsmote,
                    method = "rf",
                    verbose = FALSE,
                    metric = "ROC",
                    trControl = ctrl)
saveRDS(blsmote_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/blsmote_rf_fit2.rda")
#blsmote_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/blsmote_rf_fit2.rda")
(rfImp_blsmote <-varImp(blsmote_fit,scale = F))
summary(rfImp_blsmote$importance)
#Min.      1st Qu.  Median    Mean      3rd Qu.    Max
#0.6675  12.7192    28.8243    33.2213  50.5995   98.2614 
plot(rfImp_blsmote, top = 20)


pred.rf_blsmote<-predict(blsmote_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_blsmote[,'success'])
#precision: 0.690
#recall: 0.841
#F: 0.379

roc.curve(testdata$c5, pred.rf_blsmote[,'success'], plotit = F)
#Area under the curve (AUC): 0.973
predictions_train<-predict(blsmote_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     489      11
#success    26      58
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9366    0.7221   0.84058      0.69048  0.75817  0.89505

########################################################################
#使用Adaptive Synthetic Sampling(ADASYN)
#根據數據分佈情況為不同的少數類樣本生成不同數量的新樣本
#訓練集
set.seed(seed)
train_adas = ADAS(traindata[,c(1:4,6:36)],traindata[,-c(1:4,6:36)])$data #k=5
colnames(train_adas)[36] <- "c5"
train_adas$c5 <-as.factor(train_adas$c5)
prop.table((table(train_adas$c5)))
#   false   success 
#0.4997905 0.5002095 

#ctrl$sampling <- NULL
#ADASYN 
adas_fit <-train(c5~.,
                 data = train_adas,
                 method = "rf",
                 verbose = FALSE,
                 metric = "ROC",
                 trControl = ctrl)
saveRDS(adas_fit, file = "/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/adas_rf_fit2.rda")
#adas_fit = readRDS("/Users/wilsonwang/Desktop/7研究題目相關/R_演算法實作/資料合併/model/adas_rf_fit2.rda")
(rfImp_adas <-varImp(adas_fit,scale = F))
summary(rfImp_adas$importance)
#Min.      1st Qu.  Median    Mean      3rd Qu.    Max
#0.4977   6.2836    12.8853    34.0850  30.1977   366.7580  
plot(rfImp_adas, top = 20)

pred.rf_adas<-predict(adas_fit,testdata,type='prob')
accuracy.meas(testdata$c5, pred.rf_adas[,'success'])
#precision: 0.667
#recall: 0.812
#F: 0.366

roc.curve(testdata$c5, pred.rf_adas[,'success'], plotit = F)
#Area under the curve (AUC): 0.967
predictions_train<-predict(adas_fit,newdata=testdata)
(result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success"))
#            Reference
#Prediction false success
#false     487      14
#success    28      55
#                 敏感度高(Sensitivity)
#Accuracy Kappa   Recall       Precision  F1       Balanced Accuracy
#0.9281    0.6827   0.79710      0.66265  0.72368  0.87137

########################################################################
#對不平衡資料不同的取樣所建置的模型方式的AUC
model_list <-list(original = orig_fit,
                  weighted = weighted_fit,
                  down = down_fit,
                  up = up_fit,
                  SMOTE = smote_fit,
                  ROSE = rose_fit,
                  BLSMOTE = blsmote_fit,
                  ADASYN = adas_fit)

#產生全部取樣法來產生AUC map
model_list_roc <-model_list %>%
  map(test_roc,data = testdata)

model_list_roc %>% map(auc)
#$original Area under the curve: 0.9798
#$weighted Area under the curve: 0.9773s
#$up Area under the curve: 0.9746
#$SMOTE Area under the curve: 0.9727
#$ROSE Area under the curve: 0.9763
#$BLSMOTE Area under the curve: 0.9733
#$ADASYN Area under the curve: 0.9666
#從AUC的結果來看，RF 對不平衡資料並沒有多大差異
#視覺化
#好的模型是在較低假陽性率時具有較高的真陽性率
results_list_roc <-list(NA)
num_mod <- 1


for(the_roc in model_list_roc){
  results_list_roc[[num_mod]] <-
    tibble(TPR = the_roc$sensitivities,
               FPR = 1- the_roc$specificities,
               model = names(model_list)[num_mod])
  if(num_mod < length(model_list_roc) ){
    num_mod <- num_mod + 1
  }
  
}

results_df_roc <- bind_rows(results_list_roc);results_df_roc$model
results_df_roc$model <- factor(results_df_roc$model,
                               levels = c("original","down","SMOTE","rose_fit","up","weighted","BLSMOTE","ADASYN"))


custom_col <- c("#000000","#009E73","#0072B2","#D55E00","#FFBFFF","#CC79A7","#BB68A7","#5527B2")
#整合AUC可視化
ggplot(aes(x = FPR, y = TPR,group = model),
       data = results_df_roc) +
  geom_line(aes(color = model), size =1) +
  scale_color_manual(values = custom_col) +
  geom_abline(intercept =0,slope =1,color = "lightgray", size = 1)+
  theme_bw(base_size = 18) + coord_fixed(1)

#各別AUC可視化
ggplot(aes(x = FPR, y = TPR,group = model),
       data = results_df_roc) +
  geom_line(aes(color = model), size =1) +
  facet_wrap(vars(model),scales="free")+
  theme(aspect.ratio = 1)
  scale_color_manual(values = custom_col) +
  geom_abline(intercept =0,slope =1,color = "gray", size = 1)+
  theme_bw(base_size = 18) + coord_fixed(1)

########################################################################
#預測:使用不同的採樣后的資料所產生RF模型
predict.rf <-function(model, data){
  predictions_train<-predict(model,newdata=data)
  result<-confusionMatrix(predictions_train,testdata$c5,mode = "everything",positive = "success")
  overall <-result$overall
  byClass <-result$byClass
  imortantInfo <- list(
                       overall["Accuracy"],
                       overall["Kappa"],
                       overall["McnemarPValue"],
                       byClass["Sensitivity"],
                       byClass["Specificity"],
                       byClass["Precision"],
                       byClass["Recall"],
                       byClass["F1"])
}
sample.models <- list(orig_fit,
                      weighted_fit,
                      down_fit,
                      up_fit,
                      smote_fit,
                      rose_fit,
                      blsmote_fit,
                      adas_fit)
  
outside_test <- lapply(sample.models, predict.rf, data = testdata)
outside_test <- lapply(outside_test, as.vector)
outside_test <- do.call("rbind", outside_test)
colnames(outside_test) <- c("Accuracy", "Kappa", "McnemarPValue","Sensitivity","Specificity","Precision","Recall","F1")
rownames(outside_test) <- c("original","weighted","down","up","SMOTE","ROSE","BLSMOTE","ADASYN")
(outside_test <- as.data.frame(outside_test))
outside_test_t <-t(outside_test)
#看出來沒有，sensitivity和specificity是條件於真實label Y的概率的。
#我們講這個叫條件概率嘛
#無論Y的真實概率是多少，都不會影響sensitivity和specificity。
#也就是說，這兩個metric是不會受imbalanced data 影響的，，

#threshold是啥子哦。這麼說吧，每個分類器作出的預測呢，
#都是基於一個probability score的。一般默認的threshold呢都是0.5，
#如果probability>0.5，那麼這個sample被模型分成正例了哈，反之則是反例

outside_resampling <- resamples(model_list)
summary(outside_resampling, metric = "ROC")
#ROC 
#             Min.   1st Qu.    Median      Mean   3rd Qu.      Max.    NA's
#original 0.9130005 0.9652743 0.9773583 0.9713755 0.9818957 0.9915966    0
#weighted 0.9179436 0.9634356 0.9746663 0.9707615 0.9805348 0.9933824    0
#down     0.8650519 0.9480969 0.9705882 0.9648097 0.9852941 1.0000000    0
#up       0.9970238 0.9995057 0.9997899 0.9996285 1.0000000 1.0000000    0
#SMOTE    0.9856649 0.9931943 0.9952863 0.9949765 0.9969383 0.9996148    0
#ROSE     0.9961064 0.9991981 0.9996786 0.9993056 1.0000000 1.0000000    0
#BLSMOTE  0.9870066 0.9948943 0.9962897 0.9959594 0.9973695 0.9994351    0
#ADASYN   0.9852591 0.9935796 0.9955164 0.9953321 0.9972602 0.9999300    0
#根據重採樣結果，可以推斷出上採樣幾乎是完美的，而下採樣的效果相對較差。上採樣表現如此出色的原因是多數類中的
#樣本被複製並且有很大的潛力同時出現在模型構建和保留集中。
#實際上，所有的抽樣方法都差不多（基於測試集）。沒有採樣的基本模型擬合的統計數據彼此相當一致
#*重採樣為 0.9713755，測試集為 0.9773583)

library(randomForest)
#變異度檢查,holdout、LKOCV 和 BOOT
#acc.measure=c("auc","precision","recall","F"),default:accuracy
ROSE.holdout <- ROSE.eval(c5 ~ ., 
                          data = traindata,
                          acc.measure = "auc",
                          learner = randomForest, 
                          K = 10,
                          method.assess = "LKOCV",
                          seed = seed)
ROSE.holdout
summary(ROSE.holdout$acc) #for BOOT
#  Min.  1st Qu.  Median   Mean  3rd Qu.  Max. 
#0.8723  0.8833  0.8853  0.8850  0.8870  0.8933 

#LKOCV created 136 subsets of size 10 and one subset of size 3
#auc:0.881

#Holdout estimate of auc: 0.885 #for holdout:AUC
########################################################################
#ROC 曲線:rose_fit
library(data.table) #for data.table
pred <- predict(rose_fit,newdata = testdata) #產生預測值
roc_data <-roc(testdata$c5,as.numeric(pred),ci=TRUE)
roc_data2 <-data.table(sensitivity = roc_data$sensitivities,
                      specificity = roc_data$specificities,
                      yorden = roc_data$sensitivities + roc_data$specificities,
                      auc = roc_data$auc,
                      auc_low = as.numeric(roc_data$ci[1]),
                      acu_up = as.numeric(roc_data$ci[2])
                      ) #ci 95%信心區間

ggplot(roc_data2,aes(x = 1-specificity,y= sensitivity))+
  geom_line(color='red')+
  geom_segment(
    aes(x = 0, y = 0, xend =1, yend =1),
    linetype = "dotted",
    color = "grey50")+
    xlab("Fase Positive Rate")+
    ylab("True Positive Rate")+
    ggtitle("ROC Curve For ROSE")+
    annotate("text",
              x=0.7, y = 0.2,
              label = paste0('AUC =',round(roc_data2$auc,3)))+
    annotate("text",
              x=0.7, y = 0.12,
              label = paste0('sensitivity =',round(roc_data2$sensitivity[which.max(roc_data2$yorden)],3)))+
    annotate("text",
              x=0.7, y = 0.05,
              label = paste0('specificity =',round(roc_data2$specificity[which.max(roc_data2$yorden)],3)))+
    theme_bw()



