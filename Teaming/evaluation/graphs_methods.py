"""Create graphs showing goodness of methods"""

import pandas as pd
import matplotlib.pyplot as plt

# import data
homedir="../data/v0_teaming/teaming_434proposals_200researchers/"
m0=pd.read_csv(homedir+"data_uc1_m0/m0_goodness_scores.csv", usecols=['goodness'])
m1=pd.read_csv(homedir+"data_uc1_m1/m1_goodness_scores.csv", usecols=['goodness'])
m2=pd.read_csv(homedir+"data_uc1_m2/m2_goodness_scores.csv", usecols=['goodness'])
m3=pd.read_csv("../data/v0_teaming/data_uc1_m3/m3_goodness_scores.csv", usecols=['goodness'])

plt.figure(figsize=(20,4))

num=500
x=range(0,num)
plt.scatter(x,list(m0['goodness'][:num]),color="red", label="M0", s=15)
plt.scatter(x,list(m1['goodness'][:num]),color="lime", label="M1", s=15)
plt.scatter(x,list(m2['goodness'][:num]),color="blue", label="M2", s=15)
plt.scatter(x,list(m3['goodness'][:num]),color="magenta", label="M3", s=15)

plt.title("Use Case 1 - Plotting Matched Teams VS. Goodness (for 500 Data Points)")
plt.xlabel("Matched Team t_i for a Proposal")
plt.ylabel("Goodness Score for Team t_i")
plt.legend(loc="upper right")

plt.savefig('images/graphs_methods.png')
plt.clf()