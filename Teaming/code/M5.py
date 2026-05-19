# Implement M5 - LLM-based team generation evaluation

# Import necessary libraries
import pandas as pd
import ast
import metrics_scorer as metrics
import numpy as np
import os 

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
    
# M5
""" 
Given LLM-based recommendations, evaluate goodness for each team. And print mean±std of the goodness scores.
The goodness score is the average of the scores for each recommendation.

M5-1: Full LLM-Based Team Generation - uses an LLM to analyze the key demands of an RFP and generate team recommendations based on optimal team dynamics. 
M5-2: User-Specified Researcher with LLM Completion - involves selecting a target researcher by the user, after which the LLM fills in the remaining team spots. 
M5-3: LLM-Based Leadership Selection with Algorithmic Team Formation uses an LLM to identify a core team member (e.g., team leader, Principal Investigator (PI), anyone in a significant ad- visory role) based on expertise and relevance to the RFP, with a strategy of “weak nudging”. The best-performing algorithm from the aforementioned algorithms (M0-M4) is then chosen to assemble the rest of the team.
"""

# M5-1: Full LLM-Based Team Generation
def m5_1(llm_df, proposal_skills_df, researcher_skills_df, proposal_mapping_df):
    # Clean and map proposal links to IDs
    proposal_skills_df['proposal_id'] = proposal_skills_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_skill_map = proposal_skills_df.set_index('proposal_id').T.to_dict()
    researcher_skill_map = researcher_skills_df.set_index('researcher_name').T.to_dict()

    # Map titles to proposal IDs
    proposal_mapping_df['proposal_id'] = proposal_mapping_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    # Normalize titles for matching
    proposal_mapping_df['title'] = proposal_mapping_df['title'].str.strip().str.lower()
    llm_df['Proposal Name'] = llm_df['Proposal Name'].str.strip().str.lower()

    # Build title → proposal_id map
    title_to_id = dict(zip(proposal_mapping_df['title'], proposal_mapping_df['proposal_id']))

    scores = []

    for _, row in llm_df.iterrows():
        title = row['Proposal Name']
        
        if not isinstance(title, str):
            continue
        
        if title.endswith(','):
            title = title[:-1]
        
        try: 
            proposal_id = title_to_id[title]
        except:
            continue
        
        # print(title, proposal_id)


        # title has a comma at the end, remove that
        
        if proposal_id is None:
            continue

        try:
            team = row['Team Recommended']
        except:
            continue

        proposal_skills_raw = proposal_skill_map.get(proposal_id)
        if proposal_skills_raw is None:
            continue

        try:
            proposal_skills = ast.literal_eval(proposal_skills_raw['skills'])
        except:
            continue

        # print(researcher_skill_map)
        team=team.split('; ')
        team = [item.lstrip() for item in team]
        # print(team)
        
        for jj in team:
            if jj not in researcher_skill_map:
                # print('check')
                continue 
        if proposal_skills is None or any(name not in researcher_skill_map for name in team):
            # print('continue')
            continue
        # print(apply_ultra_metric(proposal_skills, team, researcher_skill_map))
        try:
            score = apply_ultra_metric(proposal_skills, team, researcher_skill_map)
            scores.append(score)
        except:
            continue

    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"M5-1 Goodness: {mean_score:.4f} ± {std_score:.4f}")
        return mean_score, std_score, scores
    else:
        print("No valid teams evaluated.")
        return None, None, []
        
# M5-2: User-Specified Researcher with LLM Completion
def m5_2(directory_path, proposal_skills_df, researcher_skills_df, proposal_mapping_df):
    # Clean proposal skills
    proposal_skills_df['proposal_id'] = proposal_skills_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_skill_map = proposal_skills_df.set_index('proposal_id').T.to_dict()

    # Researcher skills
    researcher_skill_map = researcher_skills_df.set_index('researcher_name').T.to_dict()

    # Title → proposal_id mapping
    proposal_mapping_df['proposal_id'] = proposal_mapping_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_mapping_df['title'] = proposal_mapping_df['title'].str.strip().str.lower()
    title_to_id = dict(zip(proposal_mapping_df['title'], proposal_mapping_df['proposal_id']))

    scores = []

    # Loop through each file in the M5-2 directory
    for file in os.listdir(directory_path):
        if not file.endswith('.csv'):
            continue

        df = pd.read_csv(os.path.join(directory_path, file))
        df['Proposal Name'] = df['Proposal Name'].astype(str).str.strip().str.lower()

        for _, row in df.iterrows():
            title = row['Proposal Name']
            proposal_id = title_to_id.get(title)
            if proposal_id is None:
                continue

            team_raw = row['Team Recommended']
            professor = row['Professor Name'].strip()

            if not isinstance(team_raw, str):
                continue

            team = [professor] + [name.strip() for name in team_raw.split('; ')]

            proposal_skills_raw = proposal_skill_map.get(proposal_id)
            if proposal_skills_raw is None:
                continue

            try:
                proposal_skills = ast.literal_eval(proposal_skills_raw['skills'])
            except:
                continue

            if any(name not in researcher_skill_map for name in team):
                continue

            try:
                score = apply_ultra_metric(proposal_skills, team, researcher_skill_map)
                scores.append(score)
            except:
                continue

    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"M5-2 Goodness: {mean_score:.4f} ± {std_score:.4f}")
        return mean_score, std_score, scores
    else:
        print("No valid teams evaluated.")
        return None, None, []

def m5_3(anchor_df, df_m0, df_m1, df_m2, df_m3, proposal_skills_df, researcher_skills_df, proposal_mapping_df):
    # Clean proposal skills
    proposal_skills_df['proposal_id'] = proposal_skills_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_skill_map = proposal_skills_df.set_index('proposal_id').T.to_dict()
    researcher_skill_map = researcher_skills_df.set_index('researcher_name').T.to_dict()

    # Map titles to proposal IDs
    proposal_mapping_df['proposal_id'] = proposal_mapping_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_mapping_df['title'] = proposal_mapping_df['title'].str.strip().str.lower()
    anchor_df['Proposal Name'] = anchor_df['Proposal Name'].str.strip().str.lower()
    title_to_id = dict(zip(proposal_mapping_df['title'], proposal_mapping_df['proposal_id']))

    # Prepare M0–M3
    def prep(df):
        df = df.copy()
        df['goodness'] = df['goodness'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and pd.notna(x) else []
        )
        df['avg_goodness'] = df['goodness'].apply(
            lambda x: np.mean(x) if x else np.nan
        )
        return df

    df_m0 = prep(df_m0)
    df_m1 = prep(df_m1)
    df_m2 = prep(df_m2)
    df_m3 = prep(df_m3)

    scores = []

    for _, row in anchor_df.iterrows():
        title = row['Proposal Name']
        professor = row['Professor Recommended'].strip()
        proposal_id = title_to_id.get(title)

        if proposal_id is None:
            continue

        candidate_teams = []
        for df in [df_m0, df_m1, df_m2, df_m3]:
            subset = df[(df['proposal_id'] == proposal_id) & (df['researcher_name'] == professor)]
            if subset.empty:
                continue
            team_ids = subset['team'].unique()
            for team_id in team_ids:
                team_rows = df[(df['proposal_id'] == proposal_id) & (df['team'] == team_id)]
                if team_rows['avg_goodness'].notna().any():
                    team_score = team_rows['avg_goodness'].iloc[0]
                    candidate_teams.append((team_rows, team_score))

        if not candidate_teams:
            continue

        # Pick best team that includes the professor
        best_team_rows, _ = max(candidate_teams, key=lambda x: x[1])
        team_names = best_team_rows['researcher_name'].tolist()

        proposal_skills_raw = proposal_skill_map.get(proposal_id)
        if proposal_skills_raw is None:
            continue
        try:
            proposal_skills = ast.literal_eval(proposal_skills_raw['skills'])
        except:
            continue

        if any(name not in researcher_skill_map for name in team_names):
            continue

        try:
            score = apply_ultra_metric(proposal_skills, team_names, researcher_skill_map)
            scores.append(score)
        except:
            continue

    if scores:
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"M5-3 Goodness: {mean_score:.4f} ± {std_score:.4f}")
        return mean_score, std_score, scores
    else:
        print("No valid teams evaluated.")
        return None, None, []

    
def main():
    # Read the LLM data 
    #home_dir = '../../2025-03-01-llm/llm_recommendations/chatgpt_m5-'
    home_dir = '../../2025-03-01-llm/llm_recommendations/chatgpt_m5-'
    home_dir2 = '../data/v0_output_teaming/teaming_100proposals_200researchers/'
    m51_data = '1/208teams_chatgpt_proposals_team_recommendations_1.csv'
    m52_dir = '2/'
    m53_data = '3/100RFPs_chatgpt_proposals_professor_recommendations.csv'

    # Load proposal and researcher skills
    proposal_skills_df = pd.read_csv(home_dir2 + 'data_uc1_m1/m1_proposal_skills.csv')
    researcher_skills_df = pd.read_csv(home_dir2 + 'data_uc1_m1/m1_researcher_skills.csv')

    # Load LLM recommendations
    m51_df = pd.read_csv(home_dir + m51_data)
    proposal_mapping_df = pd.read_csv('../data/v0_input_files/V0_proposal_links_title_synopsis.csv')
    m5_1(m51_df, proposal_skills_df, researcher_skills_df, proposal_mapping_df)

    m5_2(home_dir + m52_dir, proposal_skills_df, researcher_skills_df, proposal_mapping_df)


    # M0–M3 team data for M5-3
    df_m0 = pd.read_csv(home_dir2 + 'teaming_uc1_m0.csv')
    df_m1 = pd.read_csv(home_dir2 + 'teaming_uc1_m1.csv')
    df_m2 = pd.read_csv(home_dir2 + 'teaming_uc1_m2.csv')
    df_m3 = pd.read_csv(home_dir2 + 'teaming_uc1_m3.csv')

    # M5-3
    m53_df = pd.read_csv(home_dir + m53_data)
    m5_3(m53_df, df_m0, df_m1, df_m2, df_m3, proposal_skills_df, researcher_skills_df, proposal_mapping_df)
if __name__ == "__main__":
    main()