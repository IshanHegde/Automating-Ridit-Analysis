# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:49:26 2021

@author: ishan
"""

#_______/Import packages/______#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats

#_______/Load data/________#

df = pd.read_csv(r"C:\Users\ishan\Desktop\Fall\MATH4993\Project\Project_4\wiki4HE.csv") 

df.head()

#_______/Data preprocessing/________#

df.replace('?',np.nan,inplace=True)

df.isna().sum()



item_attributes= df[['AGE', 'GENDER', 'DOMAIN', 'PhD', 'YEARSEXP', 'UNIVERSITY', 'USERWIKI','UOC_POSITION']]

questions = df.drop(['AGE', 'GENDER', 'DOMAIN', 'PhD', 'YEARSEXP', 'UNIVERSITY', 'USERWIKI','UOC_POSITION'],axis=1)


pd.unique(item_attributes['DOMAIN'].dropna())
len(pd.unique(item_attributes['AGE']))
pd.unique(item_attributes['UOC_POSITION'].dropna())


rows_with_nan = []
for index, row in pd.DataFrame(item_attributes['YEARSEXP']).iterrows():
    is_nan_series = row.isnull()
    if is_nan_series.any():
        rows_with_nan.append(index)

rows_with_nan
rows_without_nan = pd.DataFrame(item_attributes['YEARSEXP']).dropna().index

est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
age=est.fit_transform(np.asarray(item_attributes['AGE']).reshape(-1,1))

est1 = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
exp = est1.fit_transform(np.asarray(item_attributes['YEARSEXP'].dropna()).reshape(-1,1))

exp1 = pd.DataFrame(exp,index=rows_without_nan).reindex(df.index).fillna(0)
exp2=pd.DataFrame(np.zeros((23,1)),index=rows_with_nan).reindex(df.index).replace(0,-1).fillna(0)
exp3= exp1+exp2

item_attributes['AGE']= age

item_attributes['YEARSEXP']=exp3

agerange=est.bin_edges_[0]

ager=[]
for i in range(len(agerange)-1):
    ager.append(str(round(agerange[i]))+' -- '+str(round(agerange[i+1])))

ager

exprl= est1.bin_edges_[0]
expr=[]
expr.append(str(-1)+' -- '+str(round(exprl[0])))
for i in range(len(exprl)-1):
    expr.append(str(round(exprl[i]))+' -- '+str(round(exprl[i+1])))
expr

item_attributes['UOC_POSITION'].fillna(-1,inplace=True)

plt.hist(np.asarray(item_attributes['UOC_POSITION']).astype(int))

item_attributes[item_attributes['UOC_POSITION']=='6'].shape[0]
item_attributes['UOC_POSITION'].replace('2','1',inplace=True)
item_attributes['UOC_POSITION'].replace('5','3',inplace=True)
item_attributes['UOC_POSITION'].replace('4','3',inplace=True)


att_names = {'AGE':ager,'GENDER':['MALE','FEMALE'],'DOMAIN':['','Arts & Humanities','Sciences','Health Sciences','Engineering & Architecture','Law & Politics','Unknown Domain'],'PhD':['NO','YES'],'YEARSEXP':expr,'UNIVERSITY':['','UOC','UPF'],'USERWIKI':['No','Yes'],'UOC_POSITION':{ -1: 'Unknown',1:'Professor/Associate',3:'Assistant/Lecturer',6:'Adjunct'}}


#________/Ridit Analysis/________#


def ridit_mean(ridit_prob,df):
    total = df.shape[0]
    
    val=0
    for i in range(5):
        
        val+=ridit_prob[i]*(df[df.iloc[:,1]==str(i+1)].shape[0])/total
    
    return val

def ridit_var(ridit_prob,mean,df):
    total =df.shape[0]
    val=0
    for i in range(5):
        
        val+=(ridit_prob[i]-mean)**2*(df[df.iloc[:,1]==str(i+1)].shape[0])
    
    return np.sqrt(val/(total*(total-1)))

def ridit_t(mean,var,df):
    return (0.5-mean)/var


#The results are in form of a list of tuples

def compute_result(item_attributes,questions):
    
    results=[]
    
    for n in range(item_attributes.shape[1]):
    
        re=[]
        for i in range(questions.shape[1]):
            #print(item_attributes.columns[n])
            temp = pd.concat([item_attributes.iloc[:,n],questions.iloc[:,i]],axis=1)
            temp.dropna(inplace=True)
            total = temp.shape[0]
            unique = pd.unique(temp.iloc[:,0])
            n_unique = len(unique)
            
            probs =[]
            
            for k in range(1,6):
        
                probs.append(temp[temp.iloc[:,1]==str(k)].shape[0]/total)
    
            ridit_prob=[]
    
            for j in range(5):
        
                ridit_prob.append(np.sum(probs[:j])+0.5*probs[j])
        
            result=[]
            for m in range(n_unique):
                temp1 = temp[temp.iloc[:,0]==unique[m]]
                temp1_ridit = ridit_mean(ridit_prob,temp1)
                temp1_var = ridit_var(ridit_prob,temp1_ridit,temp1)
                temp1_t= ridit_t(temp1_ridit,temp1_var,temp1)
                #print(unique[m])
                result.append((questions.columns[i],att_names[item_attributes.columns[n]][int(unique[m])],unique[m],temp1_ridit,temp1_var,temp1_t))
            
            
            re.append((item_attributes.columns[n],result))
            
        results.append(re)
        
    return results

results=[]
for n in range(item_attributes.shape[1]):
    
    re=[]
    for i in range(questions.shape[1]):
        #print(item_attributes.columns[n])
        temp = pd.concat([item_attributes.iloc[:,n],questions.iloc[:,i]],axis=1)
        temp.dropna(inplace=True)
        total = temp.shape[0]
        unique = pd.unique(temp.iloc[:,0])
        n_unique = len(unique)
        
        probs =[]
    
        for k in range(1,6):
        
            probs.append(temp[temp.iloc[:,1]==str(k)].shape[0]/total)
    
        ridit_prob=[]
    
        for j in range(5):
        
            ridit_prob.append(np.sum(probs[:j])+0.5*probs[j])
        
        result=[]
        for m in range(n_unique):
            temp1 = temp[temp.iloc[:,0]==unique[m]]
            temp1_ridit = ridit_mean(ridit_prob,temp1)
            temp1_var = ridit_var(ridit_prob,temp1_ridit,temp1)
            temp1_t= ridit_t(temp1_ridit,temp1_var,temp1)
            #print(unique[m])
            result.append((questions.columns[i],att_names[item_attributes.columns[n]][int(unique[m])],unique[m],temp1_ridit,temp1_var,temp1_t))
            
            
        re.append((item_attributes.columns[n],result))
    results.append(re)
    
    




#Find statistically significant differences

def interesting(results,sig_level):
    
    interesting=[]
    
    
    
    for i in range(len(results)):
        # Item attribute level
        
        for j in range(len(results[i])):
            #Question Level
            
            for k in range(len(results[i][j][1])):
                # Unique attribute class Level
                if np.abs(results[i][j][1][k][-1]) > stats.t.ppf(1-sig_level/2,item_attributes[item_attributes[results[i][j][0]]==results[i][j][1][k][2]].shape[0]):
                    interesting.append((results[i][j][0],results[i][j][1][k][0],results[i][j][1][k][1],results[i][j][1][k][3],results[i][j][1][k][4],results[i][j][1][k][5]))
    
    return interesting




def search_by_question(interesing,question):
    output=[]
    
    for i in range(len(a)):
        
        if a[i][1]==question:
            output.append(a[i])
    return output

def search_by_attribute(interesting,attribute):
    output=[]
    
    for i in range(len(a)):
        
        if a[i][0]==attribute:
            output.append(a[i])
    return output
    output=[]
    
    for i in range(len(a)):
        
        if a[i][0]==attribute:
            output.append(a[i])
            
    return output

a=interesting(results,0.05)

print(questions.columns)

search_by_question(a, 'Use3')

print(item_attributes.columns)

search_by_attribute(a,'USERWIKI')







