# import pandas as pd
# from collections import defaultdict
# import random
# import os

# # Import string-matching distance calculator
# from difflib import SequenceMatcher

# random.seed(12345)


# def write_file(path, data):
#     os.makedirs(os.path.split(path)[0], exist_ok=True)
#     with open(path, 'w+') as fp:
#         for d in data:
#             fp.write(f'{d}\n')

# def main():
#     path_proposals = 'proposal_skills.csv'
#     path_researchers = 'researchers_skills.csv'

#     proposals = pd.read_csv(path_proposals)
#     researchers = pd.read_csv(path_researchers)

#     pos, neg, facts = set(), set(), set()
#     # parse proposals
#     proposal_skill, skill_researcher = defaultdict(list), defaultdict(list)
#     pids = []
#     print(proposals.columns.tolist())
#     print(researchers.columns.tolist())
#     for index, proposal in proposals.iterrows():
#         pid = proposal['hyperlink'].split('/')[-1].split('.')[0].strip()
#         pids.append(pid)

#         for skill in proposal['skills'].strip(' \{\}').split(', '):
#             skill = skill.strip(' \'').replace(' ', '_')
#             facts.add(f'skill({pid},{skill}).')
#             proposal_skill[pid].append(skill)
#         # print(proposal['nsf_proposal_links_v0'], proposal['skills'])
#     # print(row['c1'], row['c2'])

#     # parse researchers
#     researcher_names = []
#     for index, researcher in researchers.iterrows():
#         name = researcher['researcher_name'].strip().replace(',', '').replace('.', '')\
#                                             .replace(' ', '_').replace('-', '_').lower()
#         researcher_names.append(name)
#         for interest in researcher['skills'].strip(' \{\}').split(', '):
#             interest = interest.strip(' \'').replace(' ', '_')
#             facts.add(f'interest({name},{interest}).')
#             skill_researcher[interest].append(name)
#         # print(researcher)

#     print("parse done")
#     # generate pos and neg
#     for pid in proposal_skill:
#         for skill in proposal_skill[pid]:
#             for key in skill_researcher.keys():
#                 # use matching threshold to check if a researcher's skill is matched with each of the proposal's one
#                 if SequenceMatcher(None, skill, key).ratio() >= 0.9:
#                     for name in skill_researcher[key]:
#                         pos.add(f'team({pid},{name}).')
#             #for name in skill_researcher[skill]:
#             #    pos.add(f'team({pid},{name}).')
#     print("pos done", len(pos))
#     num_pos = len(pos)
#     num_neg = 1 * num_pos

#     print("neg start", num_neg)
    
#     count=0
#     while len(neg) < num_neg:
#         count+=1
#         pid = random.choice(pids)
#         name = random.choice(researcher_names)
#         neg.add(f'team({pid},{name}).')
        
#         if count>num_neg:
#             break
        
#     print("neg done")
#     # split to train and test data
#     pos = list(pos)
#     neg = list(neg)
#     random.shuffle(pos)
#     random.shuffle(neg)

#     num_pos_train = int(0.8*num_pos)
#     num_neg_train = int(0.8*num_neg)

#     train_pos = pos[:num_pos_train]
#     test_pos = pos[num_pos_train:]

#     train_neg = neg[:num_neg_train]
#     test_neg = neg[num_neg_train:]

#     print("mumbo done")
#     # write data to files
#     write_file('train/train_pos.txt', train_pos)
#     write_file('train/train_neg.txt', train_neg)
#     write_file('train/train_facts.txt', facts)
#     write_file('test/test_pos.txt', test_pos)
#     write_file('test/test_neg.txt', test_neg)
#     write_file('test/test_facts.txt', facts)


# if __name__ == '__main__':
#     main()

import argparse
import pandas as pd
from collections import defaultdict
import random
import os
import re
from difflib import SequenceMatcher

DEFAULT_RANDOM_SEED = 12345
DEFAULT_NEGATIVE_MULTIPLIER = 1.0
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SKILL_MATCH_THRESHOLD = 0.9

random.seed(DEFAULT_RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare BoostSRL train/test files for IITR Teaming.")
    parser.add_argument("--proposals-file", default="proposal_skills.csv")
    parser.add_argument("--researchers-file", default="researchers_skills.csv")
    parser.add_argument("--negative-multiplier", type=float, default=DEFAULT_NEGATIVE_MULTIPLIER)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--skill-match-threshold", type=float, default=DEFAULT_SKILL_MATCH_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()

def sanitize_token(s):
    """
    Normalizes strings to be safe for logic programming parsers:
    - Lowercase, replaces spaces/hyphens with underscores
    - Removes non-alphanumeric characters
    - Handles nulls and empty strings
    """
    if s is None or pd.isna(s):
        return 'unknown'
    s = str(s).strip().strip(" '{}'\"").lower()
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'^_+', '', s) # Remove leading underscores
    return s if s != '' else 'unknown'

def write_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w+') as fp:
        for d in data:
            fp.write(f'{d}\n')

def main():
    args = parse_args()
    random.seed(args.seed)

    path_proposals = args.proposals_file
    path_researchers = args.researchers_file

    proposals = pd.read_csv(path_proposals)
    researchers = pd.read_csv(path_researchers)

    pos, neg, facts = set(), set(), set()
    proposal_skill = defaultdict(list)
    skill_researcher = defaultdict(list)
    pids = []

    # 1. Parse Proposals
    for _, row in proposals.iterrows():
        # Extract PID from hyperlink/URL field
        raw_url = row.get('hyperlink', '')
        pid = sanitize_token(raw_url.split('/')[-1].split('.')[0])
        pids.append(pid)

        skills_raw = str(row.get('skills', ''))
        # Split by comma and clean each skill
        for skill in [s for s in re.split(r',\s*', skills_raw) if s.strip()]:
            clean_s = sanitize_token(skill)
            facts.add(f'skill({pid},{clean_s}).')
            proposal_skill[pid].append(clean_s)

    # 2. Parse Researchers
    researcher_names = []
    for _, row in researchers.iterrows():
        name = sanitize_token(row.get('researcher_name', ''))
        researcher_names.append(name)

        interests_raw = str(row.get('skills', ''))
        for interest in [i for i in re.split(r',\s*', interests_raw) if i.strip()]:
            clean_i = sanitize_token(interest)
            facts.add(f'interest({name},{clean_i}).')
            skill_researcher[clean_i].append(name)

    print(f"Parse done. Found {len(pids)} proposals and {len(researcher_names)} researchers.")

    # 3. Generate Positive Examples (Fuzzy Match)
    # Uses configurable fuzzy-match threshold for building training positives.
    for pid, p_skills in proposal_skill.items():
        for s_skill in p_skills:
            for r_skill in skill_researcher.keys():
                if SequenceMatcher(None, s_skill, r_skill).ratio() >= args.skill_match_threshold:
                    for r_name in skill_researcher[r_skill]:
                        pos.add(f'team({pid},{r_name}).')

    print("Positive examples generated:", len(pos))
    
    # 4. Generate Negative Examples
    num_pos = len(pos)
    num_neg = max(1, int(round(args.negative_multiplier * num_pos)))
    
    attempts = 0
    while len(neg) < num_neg and attempts < (num_neg * 10):
        attempts += 1
        pid = random.choice(pids)
        name = random.choice(researcher_names)
        pair = f'team({pid},{name}).'
        
        # Ensure negative is not actually a positive
        if pair not in pos:
            neg.add(pair)

    print(f"Negative examples generated: {len(neg)}")

    # 5. Split and Save
    pos_list, neg_list = list(pos), list(neg)
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    split_idx_p = int(args.train_ratio * len(pos_list))
    split_idx_n = int(args.train_ratio * len(neg_list))

    # Write files
    write_file('train/train_pos.txt', pos_list[:split_idx_p])
    write_file('train/train_neg.txt', neg_list[:split_idx_n])
    write_file('train/train_facts.txt', sorted(list(facts)))
    
    write_file('test/test_pos.txt', pos_list[split_idx_p:])
    write_file('test/test_neg.txt', neg_list[split_idx_n:])
    write_file('test/test_facts.txt', sorted(list(facts)))

    print(
        "Data export complete.",
        {
            "negative_multiplier": args.negative_multiplier,
            "train_ratio": args.train_ratio,
            "skill_match_threshold": args.skill_match_threshold,
            "seed": args.seed,
            "train_pos": len(pos_list[:split_idx_p]),
            "train_neg": len(neg_list[:split_idx_n]),
            "test_pos": len(pos_list[split_idx_p:]),
            "test_neg": len(neg_list[split_idx_n:]),
        },
    )

if __name__ == '__main__':
    main()
