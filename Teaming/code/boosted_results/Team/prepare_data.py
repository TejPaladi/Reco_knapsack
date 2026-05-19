import pandas as pd
from collections import defaultdict
import random
import os

random.seed(12345)

def write_file(path, data):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as fp:
        for d in data:
            fp.write(f'{d}\n')

def main():
    path_proposals = 'proposal_skills.xlsx'
    path_researchers = 'researchers_skills.xlsx'

    proposals = pd.read_excel(path_proposals)
    researchers = pd.read_excel(path_researchers)

    pos, neg, facts = set(), set(), set()
    # parse proposals
    proposal_skill, skill_researcher = defaultdict(list), defaultdict(list)
    pids = []
    for index, proposal in proposals.iterrows():
        pid = proposal['nsf_proposal_links_v0'].split('/')[-1].split('.')[0].strip()
        pids.append(pid)

        for skill in proposal['skills'].strip(' \{\}').split(', '):
            skill = skill.strip(' \'').replace(' ', '_')
            facts.add(f'skill({pid},{skill}).')
            proposal_skill[pid].append(skill)
        # print(proposal['nsf_proposal_links_v0'], proposal['skills'])
    # print(row['c1'], row['c2'])

    # parse researchers
    researcher_names = []
    for index, researcher in researchers.iterrows():
        name = researcher['researcher_name'].strip().replace(',', '').replace('.', '')\
                                            .replace(' ', '_').replace('-', '_').lower()
        researcher_names.append(name)

        for interest in researcher['skills'].strip(' \{\}').split(', '):
            interest = interest.strip(' \'').replace(' ', '_')
            facts.add(f'interest({name},{interest}).')
            skill_researcher[interest].append(name)
        # print(researcher)

    # generate pos and neg
    for pid in proposal_skill:
        for skill in proposal_skill[pid]:
            for name in skill_researcher[skill]:
                pos.add(f'team({pid},{name}).')

    num_pos = len(pos)
    num_neg = 2 * num_pos

    while len(neg) < num_neg:
        pid = random.choice(pids)
        name = random.choice(researcher_names)
        neg.add(f'team({pid},{name}).')
        print(len(neg))

    # split to train and test data
    pos = list(pos)
    neg = list(neg)
    random.shuffle(pos)
    random.shuffle(neg)

    num_pos_train = int(0.8*num_pos)
    num_neg_train = int(0.8*num_neg)

    train_pos = pos[:num_pos_train]
    test_pos = pos[num_pos_train:]

    train_neg = neg[:num_neg_train]
    test_neg = neg[num_neg_train:]

    # write data to files
    write_file('train/train_pos.txt', train_pos)
    write_file('train/train_neg.txt', train_neg)
    write_file('train/train_facts.txt', facts)
    write_file('test/test_pos.txt', test_pos)
    write_file('test/test_neg.txt', test_neg)
    write_file('test/test_facts.txt', facts)


if __name__ == '__main__':
    main()
