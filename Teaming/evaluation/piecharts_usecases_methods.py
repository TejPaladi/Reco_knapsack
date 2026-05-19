import matplotlib.pyplot as plt
import pandas as pd 

# file paths for use case responses extracted from the human study survey
path_relevant_responses = "../data/human_study_responses/relevant_responses.csv"

# import csv
csv_relevant = pd.read_csv(path_relevant_responses, header=None, names=(
    ['relevancy_scale', 'method', 'usecase', 'comment']))

# check the frequency of each use case
usecase_freq = {}
for i in range(len(list(csv_relevant['usecase']))):
    if list(csv_relevant['usecase'])[i] not in usecase_freq:
        usecase_freq[list(csv_relevant['usecase'])[i]] = {}
    
    if list(csv_relevant['method'])[i] not in usecase_freq[list(csv_relevant['usecase'])[i]]:
        usecase_freq[list(csv_relevant['usecase'])[i]][list(csv_relevant['method'])[i]]=1
    else:
        usecase_freq[list(csv_relevant['usecase'])[i]][list(csv_relevant['method'])[i]]+=1
    
print(usecase_freq)

# Make data: I have 3 groups and 4 subgroups for each
group_names=['UC1', 'UC2', 'UC3']
group_size=[37, 27, 30]
subgroup_names=['M0', 'M1', 'M2', 'M3']*len(group_names)
subgroup_size=[8,6,18,6,5,4,13,6,10,11,6,4]

# Create colors
a, b, c=[plt.cm.Reds, plt.cm.Purples, plt.cm.Greens]

# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors= [a(0.99), b(0.99), c(0.99)], shadow=True) #, autopct=lambda x: '{:.0f}'.format(x*sum(group_size)/100))
plt.setp( mypie, width=0.3, edgecolor='white')

# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1, 
labels=subgroup_names, labeldistance=0.75, colors=[a(0.8), a(0.7), 
a(0.6), a(0.5), b(0.8), b(0.7), b(0.6), b(0.5),c(0.8), c(0.7), c(0.6), c(0.5)])
plt.setp(mypie2, width=0.4, edgecolor='white')
plt.margins(0,0)

"""subgroup_names_legs=['A.1:a1desc', 'A.2:a2desc', 'A.3:a3desc', 'A.4:a4desc'
'B.1:b1desc', 'B.2:b2desc', 'B.3:b1desc', 'B.4:b2desc','C.1:c1desc', 'C.2:c2desc', 'C.3:c3desc', 
'C.4:c4desc', 'C.5:c5desc']
plt.legend(subgroup_names_legs,loc='best')
"""
# show it
save_dir="images/"
plt.savefig(save_dir+'piechart_usecases_methods.png')