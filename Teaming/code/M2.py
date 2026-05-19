# Import text-to-classification mapper library
from mapper4_main.app.main import *
import sys
sys.path.append('./mapper4-main/')
import random
import metrics_scorer as metrics

"""
Using the mapper code: given a query consisting of [research_interest, matching_threshold, tree_type], return results. 

Text to Classification Code Mapper: http://casy.cse.sc.edu/mapper/
"""

def callMapper(research_interest: str, matching_threshold: int, acm_or_jel: str):
    """Given three parameters (research interest, threshold, and ACM/JEL), retrieve the corresponding classification codes.

    Args:
        research_interest (str): research interest to search within the mapper
        matching_threshold (integer): string similarity matching percentage
        acm_or_jel (str): "acm" or "jel"

    Example:
        Input: callMapper("artificial intelligence", 65, "acm")
        Output: ['I.2'] [['ARTIFICIAL INTELLIGENCE', '']]
    """
    with app.app_context():  # run mapper code within application context (similar to tf.Session)
        initialize()
        # parameters = research_interest, matching_threshold, acm/jel
        codes, terms = getValue(research_interest, matching_threshold, acm_or_jel)
        print(codes, terms)
        return codes, terms
    
def create_teams_for_each_person(all_researchers, target_researcher, num_of_teams, defaultSizeFlag=False, defaultSize=-1):
    """Create a certain number of teams for each researcher by randomly picking the other members

    Args:
        all_researchers (list): List of all researchers (names)
        target_researcher (str): Target researcher
        num_of_teams (int): Number of teams desired for each target researcher
        defaultSizeFlag (bool): initially False; set it to True if you want to manually set the size of team
        defaultSize (int): -1 as a placeholder. Set it to a number (the size of desired team)
        
    Returns:
        teams: teams for each researcher
    """  
    if defaultSizeFlag and defaultSize==-1:
        raise Exception("Please provide valid team size!")
    
    count=0
    pseudo_count=0     # to prevent infinite loops (just as a precaution)
    teams=[]
    
    
    # for each proposal and researcher, create <num_of_teams> random teams
    while count<num_of_teams:
        # to prevent the loop from infinitely looping
        pseudo_count+=1
        if pseudo_count>num_of_teams+10:     # regardless of when this loop starts and how many teams have been filled, it will break after num_of_teams+N times 
            break
                
        # sample team of random size    
        if defaultSizeFlag:    # if the researcher provided a team size of their own
            size=defaultSize
        else:
            size=random.randint(1,4)     # if not, select a size randomly
            if size>len(all_researchers):
                size=len(all_researchers)     # if the size is greater than the number of researchers, set it to the number of researchers
            
        if size>len(all_researchers):
            size=len(all_researchers)
        print(all_researchers, size)
        team=random.sample(all_researchers, size)
        
        # filter out from accidentally selecting the same team member twice within a team
        if target_researcher in team:
            team.remove(target_researcher)
        
        # add the target_researcher as part of the team as well
        team.insert(0, target_researcher)
        
        # filter out from accidentally selecting the same team within list of already picked teams
        if team not in teams:
            teams.append(team)
            count+=1
    
    # return teams list designed for the target member
    return teams

def apply_ultra_metric(proposal_skills, team, all_researchers_skills):
    # instantiate scorer
    m = metrics.MetricScorer()

    # initialize demand skills
    m.demand=proposal_skills
    
    # initialize supply - team, list of researchers
    m.team=team
    for i in m.team:
        m.researchers[i]=all_researchers_skills[i]
    
    # set weights for metrics [redundancy, setsize, coverage, krobustness]
    m.set_new_weights([-1, -1, 1, 1])  # this is the one by default

    # apply goodness metric
    m.run_metrics()
    
    # return goodness
    return m.goodness