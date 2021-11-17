# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:49:26 2021

@author: ishan
"""

#_______/Import packages/______#

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats
import scipy
import sklearn
import time
# 

pd.set_option('mode.chained_assignment', None)


#________/Ridit Analysis/________#

class Ridit:
    
    def __init__(self,personalAttributeDf,questionsDF):
        self._personalAttributes = personalAttributeDf
        self._questions = questionsDF


    def ridit_mean(self,ridit_prob,df):
        total = df.shape[0]
        
        val=0
        for i in range(5):
        
            val+=ridit_prob[i]*(df[df.iloc[:,1]==str(i+1)].shape[0])/total
    
        return val

    def ridit_var(self,ridit_prob,mean,df):
        total =df.shape[0]
        val=0
        for i in range(5):
        
            val+=(ridit_prob[i]-mean)**2*(df[df.iloc[:,1]==str(i+1)].shape[0])
    
        return np.sqrt(val/(total*(total-1)))


    def ridit_conf(self,mean,df,alpha):
        
        return stats.norm.ppf(1-alpha/2)/np.sqrt(12*df.shape[0])

    def ridit_W(self,mean,df):
    
        return 12*df.shape[0]*(mean-0.5)**2

    def pval(self,W,n):
        return 1-stats.chi2.cdf(W,n-1)

    #The results are in form of a list of tuples

    def results(self,att_names):
        
        results=[]
        item_attributes = self._personalAttributes
        questions = self._questions
        
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
        
                    probs.append(temp.loc[temp.iloc[:,1]==str(k)].shape[0]/total)
    
                ridit_prob=[]
    
                for j in range(5):
        
                    ridit_prob.append(np.sum(probs[:j])+0.5*probs[j])
        
                result=[]
                w=0
                for m in range(n_unique):
                    
                    temp1 = temp.loc[temp.iloc[:,0]==unique[m]]
                    temp1_ridit = self.ridit_mean(ridit_prob,temp1)
            
            
                    w+=self.ridit_W(temp1_ridit,temp1)
                    interval= self.ridit_conf(temp1_ridit,temp1,0.05)
                    #print(unique[m])
                    result.append((questions.columns[i],att_names[item_attributes.columns[n]][int(unique[m])],unique[m],round(temp1_ridit-interval,5),round(temp1_ridit,5),round(temp1_ridit+interval,5)))
            
            
                re.append((item_attributes.columns[n],result,round(w,5),round(self.pval(w,n_unique),5)))
            results.append(re)
        return results
    
    #Find statistically significant differences

    def interesting(self,results,sig_level):
    
        interesting=[]

        for i in range(len(results)):
            # Item attribute level
        
            for j in range(len(results[i])):
                #Question Level
                if(results[i][j][-1]<sig_level):
                    interesting.append(results[i][j])
            
            return interesting


    def search_by_question(self,interesting,question):
        output=[]
    
        for i in range(len(interesting)):
        
            if interesting[i][1][0][0]==question:
                output.append(interesting[i])
        return output

    def search_by_attribute(self,interesting,attribute):
        output=[]
    
        for i in range(len(interesting)):
        
            if interesting[i][0]==attribute:
                output.append(interesting[i])
        return output


def main():
    start = time.time()
    # Read the dataset
    
    df = pd.read_csv(r"C:\Users\ishan\Desktop\Fall\MATH4993\Project\Project_4\wiki4HE.csv") 
    
    df.replace('?',np.nan,inplace=True)
    
    personalAttributes= df[['AGE', 'GENDER', 'DOMAIN', 'PhD', 'YEARSEXP', 'UNIVERSITY', 'USERWIKI','UOC_POSITION']]

    questions = df.drop(['AGE', 'GENDER', 'DOMAIN', 'PhD', 'YEARSEXP', 'UNIVERSITY', 'USERWIKI','UOC_POSITION'],axis=1)
    
    rows_with_nan = []
    for index, row in pd.DataFrame(personalAttributes['YEARSEXP']).iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)

    rows_with_nan
    rows_without_nan = pd.DataFrame(personalAttributes['YEARSEXP']).dropna().index
    
    #Descritizing numeric attributes appropriately 
    
    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    age=est.fit_transform(np.asarray(personalAttributes['AGE']).reshape(-1,1))

    est1 = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
    exp = est1.fit_transform(np.asarray(personalAttributes['YEARSEXP'].dropna()).reshape(-1,1))

    exp1 = pd.DataFrame(exp,index=rows_without_nan).reindex(df.index).fillna(0)
    exp2=pd.DataFrame(np.zeros((23,1)),index=rows_with_nan).reindex(df.index).replace(0,-1).fillna(0)
    exp3= exp1+exp2
    
    personalAttributes['AGE']= age
    
    personalAttributes['YEARSEXP']=exp3
    agerange=est.bin_edges_[0]

    ager=[]
    for i in range(len(agerange)-1):
        ager.append(str(round(agerange[i]))+' -- '+str(round(agerange[i+1])))

    exprl= est1.bin_edges_[0]
    expr=[]
    expr.append(str(-1)+' -- '+str(round(exprl[0])))
    for i in range(len(exprl)-1):
        expr.append(str(round(exprl[i]))+' -- '+str(round(exprl[i+1])))
    
    personalAttributes['UOC_POSITION'].fillna(-1,inplace=True)
    personalAttributes[personalAttributes['UOC_POSITION']=='6'].shape[0]
    personalAttributes['UOC_POSITION'].replace('2','1',inplace=True)
    personalAttributes['UOC_POSITION'].replace('5','3',inplace=True)
    personalAttributes['UOC_POSITION'].replace('4','3',inplace=True)

    # A dictionary of attribute names and their categorical names    
    
    
    att_names = {'AGE':ager,'GENDER':['MALE','FEMALE'],'DOMAIN':['','Arts & Humanities','Sciences','Health Sciences','Engineering & Architecture','Law & Politics','Unknown Domain'],'PhD':['NO','YES'],'YEARSEXP':expr,'UNIVERSITY':['','UOC','UPF'],'USERWIKI':['No','Yes'],'UOC_POSITION':{ -1: 'Unknown',1:'Professor/Associate',3:'Assistant/Lecturer',6:'Adjunct'}}
    
    ridit = Ridit(personalAttributes,questions)
    
    # reuslt format: Personal attribute, Question, attribute class, Value of attribute class, Lower bound, Ridit mean, Upper bound, W-statistic, p-value
    
    # W-statistic and P-values are for the attribute in whole but Ridit mean and the bounds are for the indivudual attribute class
    
    results = ridit.results(att_names)
    
    
    
    interesting = ridit.interesting(results,0.01)
    
    #print(interesting)
    print(ridit.search_by_question(interesting, 'PU1'))
    
    #print(ridit.search_by_attribute(interesting,'AGE'))
    
    #print(interesting)
    end = time.time()
    print("Elapsed Time = %s" % (end - start))


if __name__ =="__main__":
    
    main()





