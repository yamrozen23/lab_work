
  
```{r}
library(tidyverse)

all_feats <- read_csv("C:/Users/yamro/Downloads/train_feats12.csv")


library(BayesFactor)

longer_feat <- pivot_longer(select(all_feats,-workerId,-text,-split,-startPerson,-advadj,-GI_rcloss),cols=setdiff(colnames(all_feats)[4:(ncol(all_feats)-1)],c('startPerson','advadj','GI_rcloss')))

#wider_feat <- pivot_wider(longer_feat,names_from = 'mem_type',values_from = 'value')
tz <- longer_feat %>% group_by(name) %>% 
  summarise(t=t.test(value~mem_type,var.equal=T,paired=T)$statistic,
            df=t.test(value~mem_type,var.equal=T,paired=T)$parameter,
            p=t.test(value~mem_type,var.equal=T,paired=T)$p.value,
            d = effectsize::t_to_d(t.test(value~mem_type,var.equal=T,paired=T)$statistic,t.test(value~mem_type,var.equal=T,paired=T)$parameter,paired=T)$d)

include <- tz$name[tz$p<0.05]

now_train_all <- all_feats %>% filter(split%in%c('train','valid')) %>% 
  select(-workerId,-text,-split,-advadj,-GI_rcloss)
now_test_all <- all_feats %>% filter(split=='test') %>% 
  select(-workerId,-text,-split,-advadj,-GI_rcloss)


now_train <- all_feats %>% filter(split%in%c('train','valid')) %>% 
  select(c('mem_type','startPerson',include))
now_test <- all_feats %>% filter(split=='test') %>% 
  select(c('mem_type','startPerson',include))


library(caret)
TRC <- trainControl(method='cv',5)

model_func <- function(i=1,tr_dat,ts_dat,tr_method,y_col='mem_type',pos='img',n_cv=5){
  require(caret)
  form1 <- formula(paste0('mem_type ~ (.)^',i))
  if(i==1){
    form1 <- formula('mem_type ~ .')
  }
  comb <- data.frame(model=tr_method,interaction=i,metric = c("accuracy","acc_upper","acc_lower","sens","spec"),
                     train=NA,test=NA)
  mytrc <- trainControl(method='cv',n_cv)
  now_g <- train(form1,tr_dat,method=tr_method,trControl=mytrc)
  jj <- confusionMatrix(predict(now_g,tr_dat),factor(tr_dat[[y_col]]),positive=pos)
  comb$train[comb$metric=='accuracy'] <- jj$overall["Accuracy"]
  comb$train[comb$metric=='acc_upper']  <- jj$overall["AccuracyUpper"]
  comb$train[comb$metric=='acc_lower'] <- jj$overall["AccuracyLower"]
  comb$train[comb$metric=='sens'] <- jj$byClass["Sensitivity"]
  comb$train[comb$metric=='spec'] <- jj$byClass["Specificity"]
  
  jj <- confusionMatrix(predict(now_g,ts_dat),factor(ts_dat[[y_col]]),positive=pos)
  comb$test[comb$metric=='accuracy'] <- jj$overall["Accuracy"]
  comb$test[comb$metric=='acc_upper']  <- jj$overall["AccuracyUpper"]
  comb$test[comb$metric=='acc_lower'] <- jj$overall["AccuracyLower"]
  comb$test[comb$metric=='sens'] <- jj$byClass["Sensitivity"]
  comb$test[comb$metric=='spec'] <- jj$byClass["Specificity"]
  return(comb)
}

rf_func <- function(i=1,tr_dat,ts_dat,y_col='mem_type',pos='img',n_cv=5,trees=500){
  require(caret)
  form1 <- formula(paste0('mem_type ~ (.)^',i))
  if(i==1){
    form1 <- formula('mem_type ~ .')
  }
  comb <- data.frame(model=paste0('rf_',trees),interaction=i,metric = c("accuracy","acc_upper","acc_lower","sens","spec"),
                     train=NA,test=NA)
  mytrc <- trainControl(method='cv',n_cv)
  now_g <- train(form1,tr_dat,method='rf',trControl=mytrc,ntree=trees)
  jj <- confusionMatrix(predict(now_g,tr_dat),factor(tr_dat[[y_col]]),positive=pos)
  comb$train[comb$metric=='accuracy'] <- jj$overall["Accuracy"]
  comb$train[comb$metric=='acc_upper']  <- jj$overall["AccuracyUpper"]
  comb$train[comb$metric=='acc_lower'] <- jj$overall["AccuracyLower"]
  comb$train[comb$metric=='sens'] <- jj$byClass["Sensitivity"]
  comb$train[comb$metric=='spec'] <- jj$byClass["Specificity"]
  
  jj <- confusionMatrix(predict(now_g,ts_dat),factor(ts_dat[[y_col]]),positive=pos)
  comb$test[comb$metric=='accuracy'] <- jj$overall["Accuracy"]
  comb$test[comb$metric=='acc_upper']  <- jj$overall["AccuracyUpper"]
  comb$test[comb$metric=='acc_lower'] <- jj$overall["AccuracyLower"]
  comb$test[comb$metric=='sens'] <- jj$byClass["Sensitivity"]
  comb$test[comb$metric=='spec'] <- jj$byClass["Specificity"]
  return(comb)
}


mod_wrap <- function(model_name,ints_n=3){
  if(grepl('rf',model_name)){
    now_trees <- as.numeric(str_sub(model_name,4,-1))
    jy <- lapply(1,rf_func,tr_dat=now_train,ts_dat=now_test,trees=now_trees)
  }
  else{
    jy <- lapply(1:ints_n,model_func,tr_dat=now_train,ts_dat=now_test,tr_method=model_name)  
  }
  
  mod_comb <- data.frame()
  for(i in 1:length(jy)){
    mod_comb <- rbind(mod_comb,jy[[i]])
  }
  return(mod_comb)
}

models <- c('glm','glmnet','xgbLinear','adaboost','svmLinear','svmRadial','rf_500','rf_100')
by_mod <- lapply(models,mod_wrap,ints_n=1)

all_comb <- data.frame()
for(i in 1:length(by_mod)){
  all_comb <- rbind(all_comb,by_mod[[i]])
}


all_comb_long <- all_comb %>% select(-interaction) %>% pivot_longer(cols=c(train,test),names_to = 'set')
all_comb_wider <- all_comb_long %>%
  pivot_wider(names_from = 'metric',values_from = 'value') %>% 
  mutate(set=factor(set,levels=c('train','test')))
all_comb_wider %>% ggplot(aes(x=set,y=accuracy,ymin=acc_upper,ymax=acc_lower,fill=set))+
  geom_errorbar()+
  geom_col()+
  facet_wrap(~model)
```