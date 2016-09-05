library(h2o)

##set the maximum memory size to 7g. If this is not set, the default values will be 1g for 32 bit java and 1/4 of awailable memory for 64 but java
h2o.init(nthreads=-1,max_mem_size='7G')  

##Upload the training data 
train<-h2o.uploadFile("../input/train.csv",destination_frame = "train.hex")

##Gain the log values for the dependent variable(Demanda_uni_equil)
train$subject<-log(train$Demanda_uni_equil+1)

##Create a partition to generate the product averages
avg<-train[train$Semana <= 5,]                          ##partition includes first 5 semana

##Create a partition to generate fit into a model                   
modelData<-train[train$Semana > 5 & train$Semana < 8,]  ##partition includes first 6,7 semana

##Create a partition for evaluation
eval<-train[train$Semana == 8,]                         ##partition includes first semana 8               

##Declare the set of independent attributes to learn from
independent<-c("Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID")

##Gain the product averages from the avg partition
groups<-h2o.group_by(data=avg,by="Producto_ID",mean("subject"))

##Include the obtained groups in a new dataset for model creation
modelDataTwo<-h2o.merge(x=modelData,y=groups,all.x = T)

##Use the median of the entire dataset for NAs
modelDataTwo$mean_subject[is.na(modelDataTwo$mean_subject)]<-0.7

evalTwo<-h2o.merge(x=eval,y=groups,all.x = T)
evalTwo$mean_subject[is.na(evalTwo$mean_subject)]<-0.7

##Default number of trees=50 unless specified otherwise
g<-h2o.gbm(
  training_frame = modelDataTwo,      
  validation_frame = evalTwo,  
  x=independent,                 
  y="subject",                   
  model_id="gbm1",                 
  learn_rate = 0.3,             
  score_tree_interval = 3,      
  sample_rate = 0.5,            
  col_sample_rate = 0.8,        
  offset_column = "mean_subject"
)

h2o.rm(train)
h2o.rm(avg)
h2o.rm(modelData)
h2o.rm(modelDataTwo)

##Upload the test dataset
test<-h2o.uploadFile("../input/test.csv",destination_frame = "test.hex")

##Merge the groups obtained from the train data to test dataset
testTwo<-h2o.merge(x=test,y=groups,all.x = T)

##If there are NAs after merging, fill them with the same constant value used to fill NAs i train dataset
testTwo$mean_subject[is.na(testTwo$mean_subject)]<-0.7
testTwo[1:5,]

##Predict the results using the model created.
p<-h2o.predict(g,testTwo)

##Since the dependent variale was originally convereted to its log values, they gave to be convereted back.
p<-exp(p)-1

submissionFrame<-h2o.cbind(test$id,p)
colnames(submissionFrame)<-c("id","Demanda_uni_equil")
h2o.exportFile(submissionFrame,path="output.csv")  