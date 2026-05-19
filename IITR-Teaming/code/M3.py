import random
import re
import metrics_scorer as metrics

# Import string-matching distance calculator
from difflib import SequenceMatcher

# Import neighbor files
from nlp_techniques import generate_N_grams

# M3 - Boosted Bandit Matching

def predicate_clean(line):
    """Replace/modify the generated facts/pos/neg text such that it's easier for boostsrl.jar to parse it

    Args:
        line (str): input string

    Returns:
        str: cleaned string
    """
    # have a dictionary to convert symbols to qwerty
    symbol_to_qwerty = {
    '‡Æ': 'A', '‡π': 'B', '‡ø': 'C', '‡≤': 'D', '‡æ': 'E', '‡µ': 'F',
    '‡•': 'G', 'à': 'H', '‡ú': 'I', '‡•ç': 'J', '‡û': 'K', '‡®': 'L',
    '‡ï': 'M', '‡Ø': 'N', '‡•ã': 'O', '‡¨': 'P', '‡•Ä': 'Q', '‡™': 'R',
    '‡∞': 'S', '‡∏': 'T', '‡‡æ': 'U', '‡Ç': 'V', '‡ï': 'W', '‡á': 'X',
    '‡è': 'Y', '‡Ü': 'Z', '‡£': '0', '‡¢': '1', '‡∞': '2', '‡§': '3'
    }
    
    # replace symbols with qwerty
    for symbol, qwerty in symbol_to_qwerty.items():
        line = line.replace(symbol, qwerty)

    # remove brackets []
    line=line.replace('[\'', '')     
    line=line.replace('\'', '')
    
    # remove parentheses inconsistencies
    line=line.replace('_(', '_')
    line=line.replace(')_', '_')
    line=line.replace('),', ',')
    line=line.replace(')-', '_')
    line=line.replace('))', ')')
    
    # if there are >1 parenthesis (even if not consecutive), remove the extra one
    # if line.count('(') > 1:
    #    line=line.replace('(', '', 1)
    
    # add 'general' as a common interest for those who do not have any
    line=line.replace(',).', ',general).')
    line=line.replace(',]).', ',general).')
    
    # remove extra parentheses at the end
    line=line.replace('))', ')')
    
    # remove roaming apostrophes and brackets
    line=line.replace('\'', '_')
    line=line.replace('[', '')
    line=line.replace(']', '')
    
    # remove spaces and other symbols with underscores for each line
    line=line.replace(' ', '_')
    line=line.replace(':', '')
    line=line.replace('"', '')
    line=line.replace('&', '')
    line=line.replace('§', '')
    line=line.replace(',_', '_')
    
    #line=line.replace('(SHRI', '_SHRI')
    #line=line.replace('(25', '_25')
    """line = re.sub(r'\(([^)]*)\)', r'\1', line)
    line=line.replace('.', ').')
    line=line.replace('interest', 'interest(')
    line=line.replace('team', 'team(')
    line=line.replace('))', ')')
    line=line.replace('((', '(')"""
    
    # if there are no opening parentheses, add one
    if line.startswith('skill') and not line.startswith('skill('):
        line=line.replace('skill', 'skill(')

    # return
    return line

def string_matching_ranking(all_researcher_skills: list, proposal_skills: str, pseudo_researcher_skills: dict, matching_threshold=1):
    """Given a string similarity matching threshold, check the similarity between each of the target researcher's skills and each word within the proposal skills (which ideally is the title/synopsis within the RFP).

    Args:
        all_researcher_skills (list): list of all researchers and their skills
        proposal_skills (list): list of proposal skills
        matching_threshold (int): matching threshold

    Returns:
        ranking: Return dictionary of researchers (for skills that meet the threshold), and in ranked order
    """
    ranking = {}

    pseudo_flag = False
    if pseudo_researcher_skills == {}:
        pseudo_flag = True
        # for each researcher
        for i in all_researcher_skills:
            pseudo_researcher_skills[i] = []

            # for each proposal skill
            for skill in proposal_skills:

                # for each researcher skill
                for k in all_researcher_skills[i]:
                    # use matching threshold to check if a researcher's skill is matched with each of the proposal's one
                    if SequenceMatcher(None, skill, k).ratio() >= matching_threshold:
                        pseudo_researcher_skills[i].append(skill)
                        break

    for i in all_researcher_skills:
        ranking[i] = len(pseudo_researcher_skills[i])/len(proposal_skills)

    # sort ranking{} by value
    ranking = dict(sorted(ranking.items(), key=lambda item: item[1]))

    # return
    if pseudo_flag:
        return ranking, pseudo_researcher_skills
    else:
        return ranking


def create_teams_for_each_person(ranking: list, target_researcher: str, num_of_teams: int, defaultSizeFlag=False, defaultSize=-1):
    """Create a certain number of teams for each researcher by using string matching to pick the other members

    Args:
        ranking (list): Return dictionary of skills that meet the threshold, and in ranked order
        target_researcher (str): Target researcher
        num_of_teams (int): Number of teams desired for each target researcher
        defaultSizeFlag (bool): initially False; set it to True if you want to manually set the size of team
        defaultSize (int): -1 as a placeholder. Set it to a number (the size of desired team)
        
    Returns:
        teams: teams for each researcher
    """
    if defaultSizeFlag and defaultSize==-1:
        raise Exception("Please provide valid team size!")

    count = 0
    pseudo_count = 0     # to prevent infinite loops (just as a precaution)
    teams = []

    # for each proposal and researcher, create <num_of_teams> random teams
    while count < num_of_teams:
        # to prevent the loop from infinitely looping
        pseudo_count += 1
        if pseudo_count > num_of_teams+10:     # regardless of when this loop starts and how many teams have been filled, it will break after num_of_teams+50 times
            break

        # sample team of random size    
        if defaultSizeFlag:    # if the researcher provided a team size of their own
            size=defaultSize
        else:
            size=random.randint(1,4)     # if not, select a size randomly
        
        # pick team based on ranking{}
        team = random.sample(list(ranking.keys())[-20:], size)

        # filter out from accidentally selecting the same team member twice within a team
        if target_researcher in team:
            team.remove(target_researcher)

        # add the target_researcher as part of the team as well
        team.insert(0, target_researcher)

        # filter out from accidentally selecting the same team within list of already picked teams
        if team not in teams:
            teams.append(team)
            count += 1

    # return teams list designed for the target member
    return teams


def apply_ultra_metric(proposal_skills, team, all_researchers_skills):
    # instantiate scorer
    m = metrics.MetricScorer()

    # initialize demand skills
    m.demand = proposal_skills

    # initialize supply - team, list of researchers
    m.team = team
    for i in m.team:
        m.researchers[i] = all_researchers_skills[i]

    # set weights for metrics [redundancy, setsize, coverage, krobustness]
    m.set_new_weights([-1, -1, 1, 1])  # this is the one by default

    # apply goodness metric
    m.run_metrics()

    # return goodness
    return m.goodness
