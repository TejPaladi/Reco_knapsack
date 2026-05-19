# Implement the M4 - ensemble method

# Import necessary libraries
import pandas as pd
import ast
import metrics_scorer as metrics
import numpy as np
"""
M4-1 - Best Method Selection - targets optimizing the match between RFP requirements and team formation methods. It applies methods M0-M3 to an RFP, evaluates each for effectiveness, and selects the method that best meets the specific demands of that RFP. The top team from this method is then chosen as the final recommendation. 

M4-2 - Best Results Across Methods - broadens the selection process by collecting the best team from each method M_i for a given RFP and then cross-evaluating these teams against each other. This ensures that the selected teams are robust against the top teams from all the other methods.
    
M4-3 - Teams Based on Top Items - focuses on creating a new team by combining the top-performing individuals from each method's recommendations. This hybrid team is then evaluated for effectiveness and alignment with the corresponding RFP's objectives.
    
M4-4 - Excluding Non-Deterministic Methods and Parameters - modifies the previous approaches by omitting M0 altogether, as it is a non-deterministic method and may introduce unpredictability into the team formation process. The intuition behind it is by excluding M0 (randomization), the formed teams are not only more precise but also more stable.
"""

# M4-1
"""
First, get the teaming datasets from M0-M3. For each method, we will have a dataset with the following columns: proposal_id, year, proposal_link, title, skills, researcher_name, team, goodness.
Then, for each RFP, get the method's teams that have the highest goodness scores. 
The final output will be a dataset with the same columns.
The teams are listed [[researcher_name, member1, member2, ...], [researcher_name, member1, member2, ...], ...].
The goodness is listed as "[goodness_score1, goodness_score2, ...]".
The final output will be a dataset with the same columns.
"""
def m4_1(df_m0, df_m1, df_m2, df_m3):
    # Ensure 'goodness' values are parsed into lists of floats
    def parse_goodness_column(df):
        df = df.copy()
        df['goodness'] = df['goodness'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )
        # Use average goodness score for comparison
        df['avg_goodness'] = df['goodness'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        return df

    # Apply parsing to each method's dataframe
    df_m0 = parse_goodness_column(df_m0)
    df_m1 = parse_goodness_column(df_m1)
    df_m2 = parse_goodness_column(df_m2)
    df_m3 = parse_goodness_column(df_m3)

    # Get unique proposal IDs
    proposal_ids = pd.concat([df_m0, df_m1, df_m2, df_m3])['proposal_id'].unique()
    
    final_teams = []

    for proposal_id in proposal_ids:
        # Collect candidate best teams from each method
        candidates = []

        for df in [df_m0, df_m1, df_m2, df_m3]:
            subset = df[df['proposal_id'] == proposal_id]
            if not subset.empty and subset['avg_goodness'].notna().any():
                best_row = subset.loc[subset['avg_goodness'].idxmax()]
                candidates.append(best_row)

        # Pick the best among all methods for this RFP
        if candidates:
            best_team = max(candidates, key=lambda row: row['avg_goodness'])
            final_teams.append(best_team)

    # Combine final selected teams into a DataFrame
    final_df = pd.DataFrame(final_teams)

    # Drop helper column
    if 'avg_goodness' in final_df.columns:
        final_df = final_df.drop(columns=['avg_goodness'])

    return final_df


# M4-2
"""
First, get the teaming datasets from M0-M3. For each method, we will have a dataset with the following columns: proposal_id, year, proposal_link, title, skills, researcher_name, team, goodness.
Then, for a given RFP, get the best team from each method. Then cross-evaluate these teams against each other to ensure that the selected teams are robust against the top teams from all the other methods.
The final output will be a dataset with the same columns.
"""
def m4_2(df_m0, df_m1, df_m2, df_m3):
    def prepare_df(df):
        df = df.copy()
        df['goodness'] = df['goodness'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )
        # Compute average team goodness
        df['avg_goodness'] = df['goodness'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        return df

    df_m0 = prepare_df(df_m0)
    df_m1 = prepare_df(df_m1)
    df_m2 = prepare_df(df_m2)
    df_m3 = prepare_df(df_m3)

    proposal_ids = pd.concat([df_m0, df_m1, df_m2, df_m3])['proposal_id'].unique()
    final_teams = []

    for proposal_id in proposal_ids:
        best_teams = []

        for df in [df_m0, df_m1, df_m2, df_m3]:
            subset = df[df['proposal_id'] == proposal_id]

            if not subset.empty and subset['avg_goodness'].notna().any():
                # Group by team, take average goodness (same for each row, but pick one row to extract score)
                teams = subset.groupby('team')
                team_avg = teams.first().reset_index()
                best_team_id = team_avg.loc[team_avg['avg_goodness'].idxmax()]['team']

                # Extract full team rows
                best_team_rows = subset[subset['team'] == best_team_id]
                best_teams.append(best_team_rows)

        # Choose the best team among the top teams from each method
        if best_teams:
            best_team_all = max(best_teams, key=lambda team: team['avg_goodness'].iloc[0])
            final_teams.append(best_team_all)

    # Combine all selected best teams into one DataFrame
    final_df = pd.concat(final_teams, ignore_index=True)
    final_df = final_df.drop(columns=['avg_goodness'], errors='ignore')

    return final_df

# M4-3
"""
First, get the teaming datasets from M0-M3. For each method, we will have a dataset with the following columns: proposal_id, year, proposal_link, title, skills, researcher_name, team, goodness.
For each RFP, get the top performing individuals from each method's recommendations.
Then, for that RFP, create a new team by combining these individuals.
The final output will be a dataset with the same columns.
"""
def m4_3(df_m0, df_m1, df_m2, df_m3, proposal_skills_df, researcher_skills_df):
    def prepare_df(df):
        df = df.copy()
        df['goodness'] = df['goodness'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )
        df['avg_goodness'] = df['goodness'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        return df

    df_m0 = prepare_df(df_m0)
    df_m1 = prepare_df(df_m1)
    df_m2 = prepare_df(df_m2)
    df_m3 = prepare_df(df_m3)

    researcher_skill_map = researcher_skills_df.set_index('researcher_name').T.to_dict()
    proposal_skill_map = proposal_skills_df.set_index('nsf_proposal_links_v0').T.to_dict()

    proposal_ids = pd.concat([df_m0, df_m1, df_m2, df_m3])['proposal_id'].unique()
    final_teams = []

    for proposal_id in proposal_ids:
        top_individuals = []

        for df in [df_m0, df_m1, df_m2, df_m3]:
            subset = df[df['proposal_id'] == proposal_id]
            if not subset.empty and subset['avg_goodness'].notna().any():
                top_person = subset.loc[subset['avg_goodness'].idxmax()]
                top_individuals.append(top_person)

        if top_individuals:
            team_df = pd.DataFrame(top_individuals)
            team_names = team_df['researcher_name'].tolist()

            proposal_skills_df['nsf_proposal_links_v0'] = proposal_skills_df['nsf_proposal_links_v0'].apply(lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x)
            proposal_skill_map = proposal_skills_df.set_index('nsf_proposal_links_v0').T.to_dict()
            
            proposal_skills = proposal_skill_map.get(proposal_id, None)
            if proposal_skills is None:
                continue  # skip if no skills found

            team_skills = {name: researcher_skill_map.get(name, None) for name in team_names}
            if any(v is None for v in team_skills.values()):
                continue  # skip if any researcher has missing skills

            # Just call the provided ultra metric function
            try:
                score = apply_ultra_metric(proposal_skills, team_names, researcher_skill_map)

                # Assign the recomputed score
                team_df['goodness'] = [score] * len(team_df)

                final_teams.append(team_df)
            except:
                continue

    final_df = pd.concat(final_teams, ignore_index=True)
    final_df = final_df[df_m0.columns]  # match original column order
    return final_df

# M4-4
"""
Select best results but without M0.
"""
def m4_4(df_m1, df_m2, df_m3, proposal_skills_df, researcher_skills_df):
    def prepare_df(df):
        df = df.copy()
        df['goodness'] = df['goodness'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
        )
        df['avg_goodness'] = df['goodness'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else np.nan
        )
        return df

    df_m1 = prepare_df(df_m1)
    df_m2 = prepare_df(df_m2)
    df_m3 = prepare_df(df_m3)

    researcher_skill_map = researcher_skills_df.set_index('researcher_name').T.to_dict()

    # Clean proposal IDs
    proposal_skills_df['nsf_proposal_links_v0'] = proposal_skills_df['nsf_proposal_links_v0'].apply(
        lambda x: x.split('/')[-1].replace('.htm', '') if isinstance(x, str) else x
    )
    proposal_skill_map = proposal_skills_df.set_index('nsf_proposal_links_v0').T.to_dict()

    proposal_ids = pd.concat([df_m1, df_m2, df_m3])['proposal_id'].unique()
    final_teams = []

    for proposal_id in proposal_ids:
        best_teams = []

        for df in [df_m1, df_m2, df_m3]:
            subset = df[df['proposal_id'] == proposal_id]
            if not subset.empty and subset['avg_goodness'].notna().any():
                # Group by team and take the best one
                grouped = subset.groupby('team').first().reset_index()
                best_team_id = grouped.loc[grouped['avg_goodness'].idxmax()]['team']
                best_team_rows = subset[subset['team'] == best_team_id]
                best_teams.append(best_team_rows)

        if best_teams:
            # Re-evaluate each best team using ultra_metric
            reevaluated_teams = []
            for team_df in best_teams:
                team_names = team_df['researcher_name'].tolist()
                proposal_skills = proposal_skill_map.get(proposal_id)
                team_skills = {name: researcher_skill_map.get(name, None) for name in team_names}

                if proposal_skills is None or any(v is None for v in team_skills.values()):
                    continue

                try:
                    score = apply_ultra_metric(proposal_skills, team_names, researcher_skill_map)
                    team_df['goodness'] = [score] * len(team_df)
                    reevaluated_teams.append(team_df)
                except:
                    continue

            # Select best reevaluated team
            if reevaluated_teams:
                best_final = max(reevaluated_teams, key=lambda x: x['goodness'].iloc[0])
                final_teams.append(best_final)

    final_df = pd.concat(final_teams, ignore_index=True)
    final_df = final_df[df_m1.columns]  # Match schema
    return final_df


def main():
    # Load the datasets from M0-M3
    home_dir= '../data/v0_output_teaming/teaming_100proposals_200researchers/'

    df_m0 = filter_first_46_researchers_from_df(pd.read_csv(home_dir + 'teaming_uc1_m0.csv'))
    df_m1 = filter_first_46_researchers_from_df(pd.read_csv(home_dir + 'teaming_uc1_m1.csv'))
    df_m2 = filter_first_46_researchers_from_df(pd.read_csv(home_dir + 'teaming_uc1_m2.csv'))
    df_m3 = filter_first_46_researchers_from_df(pd.read_csv(home_dir + 'teaming_uc1_m3.csv'))

    # Ensure 'goodness' is numeric in all dataframes
    #for df in [df_m0, df_m1, df_m2, df_m3]:
    #    df['goodness'] = pd.to_numeric(df['goodness'], errors='coerce')

    # Call the M4 methods
    final_teams_m4_1 = m4_1(df_m0, df_m1, df_m2, df_m3)
    final_teams_m4_1.to_csv(home_dir + 'teaming_uc1_m4_1.csv', index=False)
    compute_average_goodness(home_dir + 'teaming_uc1_m4_1.csv')
    
    
    final_teams_m4_2 = m4_2(df_m0, df_m1, df_m2, df_m3)
    final_teams_m4_2.to_csv(home_dir + 'teaming_uc1_m4_2.csv', index=False)
    compute_average_goodness(home_dir + 'teaming_uc1_m4_2.csv')
    
    # Load proposal and researcher skills
    proposal_skills_df = pd.read_csv(home_dir + 'data_uc1_m1/m1_proposal_skills.csv')
    researcher_skills_df = pd.read_csv(home_dir + 'data_uc1_m1/m1_researcher_skills.csv')
    final_teams_m4_3 = m4_3(
        df_m0, df_m1, df_m2, df_m3,
        proposal_skills_df, researcher_skills_df)    
    final_teams_m4_3.to_csv(home_dir + 'teaming_uc1_m4_3.csv', index=False)
    compute_average_goodness(home_dir + 'teaming_uc1_m4_3.csv')
    
    final_teams_m4_4 = m4_4(df_m1, df_m2, df_m3, proposal_skills_df, researcher_skills_df)
    final_teams_m4_4.to_csv(home_dir + 'teaming_uc1_m4_4.csv', index=False)
    compute_average_goodness(home_dir + 'teaming_uc1_m4_4.csv')
    
    print("Final teams saved to csv files.")

def filter_first_46_researchers_from_df(df_m0):
    # Get first 46 unique researchers by order of appearance
    top_46_researchers = df_m0['researcher_name'].dropna().unique()[:46]

    # Filter the DataFrame to only include those researchers
    filtered_df = df_m0[df_m0['researcher_name'].isin(top_46_researchers)].copy()

    return filtered_df


def compute_average_goodness(csv_path):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert 'goodness' strings to list of floats
    df['goodness'] = df['goodness'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else []
    )

    # Flatten all the scores into a single list
    all_scores = [score for sublist in df['goodness'] if isinstance(sublist, list) for score in sublist]

    # Compute average
    if all_scores:
        avg_score = np.mean(all_scores)
        print(f"Average goodness score: {avg_score:.4f}")
        return avg_score
    else:
        print("No valid goodness scores found.")
        return None

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
    
if __name__ == "__main__":
    main()