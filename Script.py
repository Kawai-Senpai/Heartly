# %%
import pandas as pd
import csv 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

#change datatype from object to float
df["BMI"]=df["BMI"].astype("Float32")
df["PhysicalHealth"]=df["PhysicalHealth"].astype("Float32")
df["MentalHealth"]=df["MentalHealth"].astype("Float32")
df["SleepTime"]=df["SleepTime"].astype("Float32")

df


# %%
print("Finding out all the unique values \n\n")

#unique function is used for seeing the unique elements present in this perticular column
print("HeartDisease -->",df["HeartDisease"].unique())
#sort function is used for sorting the numbers in assending order of the column and it is used in continious 
print("BMI -->",np.sort((df["BMI"].unique())))
print("AlcoholDrinking -->",df["AlcoholDrinking"].unique())
print("Stroke -->",df["Stroke"].unique())
print("PhysicalHealth -->",np.sort((df["PhysicalHealth"].unique())))
print("MentalHealth -->",np.sort((df["MentalHealth"].unique())))
print("DiffWalking -->",df["DiffWalking"].unique())
print("Sex -->",df["Sex"].unique())
print("AgeCategory	 -->",np.sort((df["AgeCategory"].unique())))
print("Race -->",df["Race"].unique())
print("Diabetic-->",df["Diabetic"].unique())
print("PhysicalActivity-->",df["PhysicalActivity"].unique())
print("GenHealth -->",df["GenHealth"].unique())
print("SleepTime -->",np.sort((df["SleepTime"].unique())))
print("Asthma -->",df["Asthma"].unique())
print("KidneyDisease -->",df["KidneyDisease"].unique())
print("SkinCancer -->",df["SkinCancer"].unique())


# %% [markdown]
# ## PROCESS DATA
# 
# 
# 
# Heart Disease-->ONE HOT ENCODING    
# 
# BMI-->CONTINIOUS
# 
# Smoking-->ONE HOT ENCODING
# 
# AlcoholDrinking-->ONE HOT ENCODING	
# 
# Stroke-->ONE HOT ENCODING
# 
# PhysicalHealth-->CONTINIOUS	
# 
# MentalHealth-->CONTINIOUS	
# 
# DiffWalking-->ONE HOT ENCODING
# 
# Sex-->ONE HOT ENCODING	
# 
# AgeCategory-->CONTINIOUS
# 
# Race-->ONE HOT ENCODING		
# 
# Diabetic-->ONE HOT ENCODING	
# 
# PhysicalActivity-->ONE HOT ENCODING		
# 
# GenHealth-->ONE HOT ENCODING	
# 
# SleepTime-->CONTINIOUS		
# 
# Asthma-->ONE HOT ENCODING		
# 
# KidneyDisease-->ONE HOT ENCODING	
# 		
# SkinCancer-->ONE HOT ENCODING		

# %%
#performing one hot encoding
ohe= pd.get_dummies(df, columns=["HeartDisease","Smoking","AgeCategory","AlcoholDrinking","Stroke","DiffWalking","Sex","Race","Diabetic","PhysicalActivity","GenHealth","Asthma","KidneyDisease","SkinCancer"])

data=ohe.drop(["HeartDisease_No","BMI","Smoking_No","AlcoholDrinking_No","Stroke_No","DiffWalking_No","Diabetic_No","PhysicalActivity_No","Asthma_No","KidneyDisease_No","SkinCancer_No"], axis=1)
data

# %%
#Correlation analysis and feature reduction

corr_limit = 0.025   #define the hight limit of the graph

corr = data.corr()
x = list((corr.columns)[:-1])
y =list((corr.iloc[-1,:]).iloc[:-1])

plt.figure(figsize=(12,5))
plt.bar(x,y)
plt.xticks(rotation=90)
plt.show()

drop = []
for i in range(len(x)):
    if(y[i]<=corr_limit and y[i]>=(-corr_limit)):
        if 'AgeCategory' in x[i]:
            continue
        drop.append(x[i])

print("Dropped columns : ",drop)
data.drop(drop,axis=1,inplace=True)
data


# %% [markdown]
# displaying reduced Dataset and showing the new graph

# %%
corr = data.corr()
x = list((corr.columns)[:-1])
y =list((corr.iloc[-1,:]).iloc[:-1])

plt.figure(figsize=(12,5))
plt.bar(x,y)
plt.xticks(rotation=90)
plt.show()

# %%
#extracting x,y
x=data.drop("HeartDisease_Yes",axis=1)
y=data[["HeartDisease_Yes"]]
x.columns
#x,y

# %% [markdown]
# ## DEEP LEARNING

# %%
import torch
import torch.nn as nn
device="cuda"

# %%
x_data=torch.tensor(x.values.astype("float32"),dtype=torch.float32,device=device)
y_data=torch.tensor(y.values,dtype=torch.float32,device=device)
x_data,y_data

# %%
#getting batch
def get_batch(batch_size):
    # Get a random index to select the batch from the data
    idx = torch.randperm(x_data.size(0))[:batch_size]
    # Get the batch tensors based on the selected index
    x_batch = x_data[idx]
    y_batch = y_data[idx]
    return x_batch, y_batch

# %%
get_batch(5)

# %%
#Building the Neural Network

class neuralnet(nn.Module):

    def __init__(self):
        super(neuralnet,self).__init__()

        self.fc1=nn.Linear(30,40)
        self.fc2=nn.Linear(40,50)
        self.fc3=nn.Linear(50,60)
        self.fc4=nn.Linear(60,70)
        self.fc5=nn.Linear(70,80)
        self.fc6=nn.Linear(80,90)
        self.fc7=nn.Linear(90,100)
        self.fc8=nn.Linear(100,110)
        self.fc9=nn.Linear(110,120)
        self.fc10=nn.Linear(120,130)
        self.fc11=nn.Linear(130,120)
        self.fc12=nn.Linear(120,110)
        self.fc13=nn.Linear(110,100)
        self.fc14=nn.Linear(100,90)
        self.fc15=nn.Linear(90,80)
        self.fc16=nn.Linear(80,70)
        self.fc17=nn.Linear(70,60)
        self.fc18=nn.Linear(60,50)
        self.fc19=nn.Linear(50,40)
        self.fc20=nn.Linear(40,30)
        self.fc21=nn.Linear(30,20)
        self.fc22=nn.Linear(20,10)
        self.fc23=nn.Linear(10,1)
        self.af=nn.GELU()
        self.sigmoid=nn.Sigmoid()
        self.dropout = nn.Dropout(0.05)

    def forward(self,x):
        
        out = self.fc1(x)
        out = self.af(out)
        out = self.fc2(out)
        out = self.af(out)
        out = self.fc3(out)

        out = self.fc4(out)
        out = self.af(out)
        out = self.fc5(out)
        out = self.af(out)
        out = self.fc6(out)
        out = self.dropout(out)
        out = self.fc7(out)
        out = self.af(out)
        out = self.fc8(out)
        out = self.af(out)
        out = self.fc9(out)
        out = self.dropout(out)
        out = self.fc10(out)
        out = self.af(out)
        out = self.fc11(out)
        out = self.af(out)
        out = self.fc12(out)
        out = self.dropout(out)
        out = self.fc13(out)
        out = self.af(out)
        out = self.fc14(out)
        out = self.af(out)
        out = self.fc15(out)
        out = self.dropout(out)
        out = self.fc16(out)
        out = self.af(out)
        out = self.fc17(out)
        out = self.af(out)
        out = self.fc18(out)
        out = self.dropout(out)
        out = self.fc19(out)
        out = self.af(out)
        out = self.fc20(out)
        out = self.af(out)
        out = self.fc21(out)
        out = self.fc22(out)
        out = self.fc23(out)

        out=self.sigmoid(out)
        return(out)


# %%
#testing out the object and compare it with the actual output

model=neuralnet().to(device=device)
x,y=get_batch(5)
y_got=model(x)
y_got,y


# %%
learning_rate = 0.001
epochs = 10000
batch_no = 319794
optimiser = torch.optim.Adam(model.parameters(),learning_rate)

losses = []
variances = []

for i in range(epochs):

    x,y=get_batch(batch_no)
    y_got=model(x)
    #Calculate Loss
    loss=nn.functional.binary_cross_entropy(y_got,y)
    loss.backward()    #d(cost)/d(w)

    #Update Weights
    optimiser.step()  #w_new=old_w-learning rate*(d(cost)/d(w))

    #Zero the gradients ater updating
    optimiser.zero_grad()  #delete calculated d(cost)/d(w)

    if(i+1)%10 == 0:
        print(f"after {i}, Loss = ", loss.item(), end = "")
        losses.append(loss.item())

        # calculate variance
        var = torch.var(torch.tensor(losses))
        print(f"Var = ", var.item())
        variances.append(var.item())

# plot losses
plt.subplot(2, 1, 1)
plt.plot(losses)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plot variances
plt.subplot(2, 1, 2)
plt.plot(variances)
plt.title('Variance of Loss')
plt.xlabel('Epoch')
plt.ylabel('Variance')

plt.tight_layout()
plt.show()

# %%
a,b=get_batch(5)
model(a),b

# %% [markdown]
# ## save

# %%
torch.save(model,'Heart_disease.CookieNeko')

# %% [markdown]
# ## load

# %%
bla = torch.load('Heart_disease.CookieNeko')
bla = bla.to(device)

x = x_data[1020].to(device)
y = y_data[1020].to(device)

p = bla(x)

print("Predicted : ",p)
print("Actual : ",y)


