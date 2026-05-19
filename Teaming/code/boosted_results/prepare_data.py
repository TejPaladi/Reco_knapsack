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
#     path_proposals = 'proposal_skills.xlsx'
#     path_researchers = 'researchers_skills.xlsx'

#     proposals = pd.read_excel(path_proposals)
#     researchers = pd.read_excel(path_researchers)

#     pos, neg, facts = set(), set(), set()
#     # parse proposals
#     proposal_skill, skill_researcher = defaultdict(list), defaultdict(list)
#     pids = []
#     for index, proposal in proposals.iterrows():
#         pid = proposal['nsf_proposal_links_v1'].split('/')[-1].split('.')[0].strip()
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
#                 if SequenceMatcher(None, skill, key).ratio() >= 0.8:
#                     for name in skill_researcher[key]:
#                         pos.add(f'team({pid},{name}).')
#             #for name in skill_researcher[skill]:
#             #    pos.add(f'team({pid},{name}).')
#     print("pos done", len(pos))
#     num_pos = len(pos)
#     # num_neg = 2 * num_pos
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
import pandas as pd
from collections import defaultdict
import random
import os
import re

# Import string-matching distance calculator
from difflib import SequenceMatcher

random.seed(12345)


def sanitize_token(s):
    """
    Normalize a token so it is safe for WILL / BoostSRL:
    - strip quotes/spaces
    - lowercase
    - replace spaces with underscores
    - remove anything that's not a-z0-9 or underscore
    - strip leading underscores
    - return 'unknown' if result is empty
    """
    if s is None:
        return 'unknown'
    s = str(s).strip()
    # remove surrounding braces/quotes/spaces often present in CSV fields
    s = s.strip(" '{}'\"")
    s = s.lower()
    s = s.replace('-', '_')
    s = s.replace(' ', '_')
    # keep only letters, digits and underscore
    s = re.sub(r'[^a-z0-9_]', '', s)
    # strip leading underscores which WILL parser dislikes
    s = re.sub(r'^_+', '', s)
    if s == '':
        return 'unknown'
    return s


def write_file(path, data):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w+') as fp:
        for d in data:
            fp.write(f'{d}\n')


def main():
    # filenames (match what you showed)
    path_proposals = 'proposal_skills.csv'
    path_researchers = 'researcher_skills.csv'

    # read CSVs
    proposals = pd.read_csv(path_proposals)
    researchers = pd.read_csv(path_researchers)

    pos, neg, facts = set(), set(), set()
    proposal_skill, skill_researcher = defaultdict(list), defaultdict(list)
    pids = []

    # parse proposals
    for index, proposal in proposals.iterrows():
        # Extract pid from the URL/filename field
        raw_pid = proposal.get('nsf_proposal_links_v1', '')
        # defensive: avoid crashing if field missing
        pid = sanitize_token(raw_pid.split('/')[-1].split('.')[0])
        pids.append(pid)

        skills_field = str(proposal.get('skills', ''))
        for skill in [s for s in re.split(r',\s*', skills_field.strip(" {}'\"")) if s != '']:
            skill_clean = sanitize_token(skill)
            facts.add(f'skill({pid},{skill_clean}).')
            proposal_skill[pid].append(skill_clean)

    # parse researchers
    researcher_names = []
    for index, researcher in researchers.iterrows():
        raw_name = researcher.get('researcher_name', '')
        name = sanitize_token(raw_name)
        researcher_names.append(name)

        skills_field = str(researcher.get('skills', ''))
        for interest in [s for s in re.split(r',\s*', skills_field.strip(" {}'\"")) if s != '']:
            interest_clean = sanitize_token(interest)
            facts.add(f'interest({name},{interest_clean}).')
            skill_researcher[interest_clean].append(name)

    print("parse done")

    # generate pos (fuzzy matching between proposal skills and researcher interests)
    for pid in proposal_skill:
        for skill in proposal_skill[pid]:
            for key in skill_researcher.keys():
                if SequenceMatcher(None, skill, key).ratio() >= 0.8:
                    for name in skill_researcher[key]:
                        pos.add(f'team({pid},{name}).')

    print("pos done", len(pos))
    num_pos = len(pos)
    num_neg = 2 * num_pos

    print("neg start", num_neg)

    # generate neg examples ensuring we don't accidentally add positives
    count = 0
    while len(neg) < num_neg:
        count += 1
        pid = random.choice(pids)
        name = random.choice(researcher_names)
        pair = f'team({pid},{name}).'
        if pair in pos:
            continue
        neg.add(pair)

        if count > num_neg * 10:
            # safety break to avoid infinite loop if space too small
            break

    print("neg done", len(neg))

    # split to train and test data
    pos_list = list(pos)
    neg_list = list(neg)
    random.shuffle(pos_list)
    random.shuffle(neg_list)

    num_pos_train = int(0.8 * num_pos)
    num_neg_train = int(0.8 * num_neg)

    train_pos = pos_list[:num_pos_train]
    test_pos = pos_list[num_pos_train:]

    train_neg = neg_list[:num_neg_train]
    test_neg = neg_list[num_neg_train:]

    print("mumbo done")
    print(f"train_pos: {len(train_pos)}, train_neg: {len(train_neg)}, test_pos: {len(test_pos)}, test_neg: {len(test_neg)}")

    # write data to files (facts sorted for deterministic order)
    write_file('train/train_pos.txt', train_pos)
    write_file('train/train_neg.txt', train_neg)
    write_file('train/train_facts.txt', sorted(facts))
    write_file('test/test_pos.txt', test_pos)
    write_file('test/test_neg.txt', test_neg)
    write_file('test/test_facts.txt', sorted(facts))

    # print short sample for quick sanity-check
    print("\nsample train_pos (first 10):")
    for x in train_pos[:10]:
        print(" ", x)
    print("\nsample test_facts (first 10):")
    for x in list(sorted(facts))[:10]:
        print(" ", x)


if __name__ == '__main__':
    main()