import pandas as pd

home_dir="../data/v1_output_teaming/teaming_1698proposals_316researchers/"

m0=home_dir+'teaming_uc1_m0.csv'
m1=home_dir+'teaming_uc1_m1.csv'
# m2=home_dir+'teaming_uc1_m2.csv'
m3=home_dir+'teaming_uc1_m3.csv'
m6 = home_dir+'teaming_uc1_m6.csv'
m7 = home_dir+'teaming_uc1_m7.csv'

m0=pd.read_csv(m0)['goodness']
m1=pd.read_csv(m1)['goodness']
# m2=pd.read_csv(m2)['goodness']
m3=pd.read_csv(m3)['goodness']
m6=pd.read_csv(m6)['goodness']
m7=pd.read_csv(m7)['goodness']

import ast

good0=[]
good1=[]
good2=[]
good3=[]
good4=[]
good5=[]

n=50
for i in m0[:n]:
    good0+=ast.literal_eval(i)

for i in m1[:n]:
    good1+=ast.literal_eval(i)

# for i in m2[:n]:
#     good2+=ast.literal_eval(i)

for i in m3[:n]:
    good3+=ast.literal_eval(i)


for i in m6[:n]:
    good4+=ast.literal_eval(i)


for i in m7[:n]:
    good5+=ast.literal_eval(i)

avg0=sum(good0)/len(good0)
avg1=sum(good1)/len(good1)
# avg2=sum(good2)/len(good2)
avg3=sum(good3)/len(good3)
avg4=sum(good4)/len(good4)
avg5=sum(good5)/len(good5)

print(avg0, avg1, avg3, avg4, avg5)