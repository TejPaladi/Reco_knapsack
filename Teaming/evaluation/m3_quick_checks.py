"""Check how many proposals/users M3 gave results for."""
# importing Statistics module
import statistics
from math import comb

# initialize
proposals={}
users={}
proposals_users={}

# read dir
home_dir='../code/boosted_results/test/'

# open results file
results=open(home_dir+'results_team.db', 'r')

# read lines
for i in results.readlines():
    split_text=i.split()
    
    # extract proposal
    proposal=split_text[0].split('(')[1]
    
    # extract user
    user=split_text[1].split(')')[0]
    
    # save
    if proposal not in proposals:
        proposals[proposal]=1
    else:
        proposals[proposal]+=1
        
    if user not in users:
        users[user]=1
    else:
        users[user]+=1
        
    if proposal not in proposals_users:
        proposals_users[proposal]=[user]
    else:
        if user not in proposals_users[proposal]:
            proposals_users[proposal].append(user)

# determine team count
# max_teams=10
# p=proposals.values()
# u=users.values()
# teams=[]
# for i in p:
#     count=i
#     try:
#         if comb(count,5) > 10:
#             teams.extend([10]*count)
#         else:
#             teams.extend(comb(count,5)*count)
            
#         if count > 10:
#             teams.extend([3]*(100-count))
#         else:
#             teams.extend([count]*(100-count))
#     except:
#         pass
#         #teams.extend([1]*len(proposals))

# print("Total proposals:", len(proposals))
# print("Total users:", len(users))

# print(sum(teams)/len(teams), statistics.stdev(teams))


# determine team count
from math import comb
import statistics

max_teams = 10
teams = []

for count in proposals.values():
    # If a proposal has fewer than 5 candidates, you can't form a team of size 5
    if count < 5:
        # choose what you want here: 0 means "no possible teams"
        teams.append(0)
        continue

    possible = comb(count, 5)

    # Cap at max_teams
    capped = min(possible, max_teams)

    # Your original logic was weirdly scaling by "count";
    # if you intended to add ONE value per proposal, do this:
    teams.append(capped)

# Safe printing
if len(teams) == 0:
    print("teams is empty — nothing to summarize.")
elif len(teams) == 1:
    print("Avg:", teams[0], "Stdev: N/A (need 2+ values)")
else:
    print("Avg:", sum(teams) / len(teams), "Stdev:", statistics.stdev(teams))


# for the missing users
#teams.extend([1]*(200-len(users)))
# print
#print(sum(users.values())/len(users))
#print(statistics.stdev(users.values()))
#print(teams)
#print(users)
#print(proposals_users)
