import plotly.graph_objects as go
import urllib
import json
import pandas as pd
import random

"""Create Sankey diagrams for responses derived from the human study."""

# file paths for relevant and useful responses extracted from the human study survey
path_relevant_responses = "../data/human_study_responses/relevant_responses.csv"
path_useful_responses = "../data/human_study_responses/useful_responses.csv"

# import csvs
csv_relevant = pd.read_csv(path_relevant_responses, header=None, names=(
    ['relevancy_scale', 'method', 'usecase', 'comment']))
csv_useful = pd.read_csv(path_useful_responses, header=None, names=(
    ['useful_scale', 'method', 'usecase', 'comment']))

# convert these two into a single file
csv_relevant_useful = csv_relevant.copy()
# add the extra column from useful_responses to existing one in relevant_responses
# csv_relevant_useful['useful_scale'] = csv_useful['useful_scale']

# determine sources and targets of information flow
sources = []
targets = []
for i in range(len(csv_relevant_useful)):
    sources.append(list(csv_relevant_useful['usecase'])[i])    # UC1 ---> M0
    targets.append(list(csv_relevant_useful['method'])[i])

for i in range(len(csv_relevant_useful)):
    sources.append(list(csv_relevant_useful['method'])[
                   i])    # UC1 ---> M0 ---> relevancy
    targets.append(list(csv_relevant_useful['relevancy_scale'])[i])

for i in range(len(csv_relevant_useful)):
    # UC1 ---> M0 ---> relevancy ---> usefulness
    sources.append(list(csv_relevant_useful['relevancy_scale'])[i])
    targets.append(list(csv_useful['useful_scale'])[i])

# values
values = [[1]*len(sources)][0]

# remove 'nan' and other irrelevant values from the data and don't store those mappings
og_indices = {}

count = 0   # keep track of indices. We're using the index of nodes instead of the actual node names
for i in sources+targets:
    if i not in og_indices.keys():
        og_indices[i] = count
        count += 1

og_indices['Will not be using it']=0     # since we received 0 responses for this one on the usefulness scale

source_indices = []    # keep track of only relevant data
target_indices = []
for i in range(len(sources)):
    if sources[i] in og_indices and targets[i] in og_indices:
        source_indices.append(og_indices[sources[i]])
        target_indices.append(og_indices[targets[i]])

# frequency counts: check how many responses are available for each node
og_counts={}
for i in og_indices.keys():
    if i.startswith("UC"):
        og_counts[i]=source_indices.count(og_indices[i])
    else:
        og_counts[i]=target_indices.count(og_indices[i])

print(og_counts)

from collections import Counter

# Collect pairs of relevancy -> usefulness transitions
relevancy_to_usefulness = list(zip(csv_relevant_useful['relevancy_scale'], csv_useful['useful_scale']))

# Count frequency of each unique transition
transition_counts = Counter(relevancy_to_usefulness)

# Print the transitions in a readable format
print("\nRelevancy → Usefulness transitions:\n")
for (rel, use), count in transition_counts.items():
    print(f"'{rel}' → '{use}': {count}")


# some of the data contains '_'. Remove that from those nodes and use titlecase on them
new_indices = []
for i in og_indices:
    if '_' in i:
        split_word = i.split('_')
        new_indices.append(' '.join(split_word).title())
    else:
        new_indices.append(i.title())
    
    if 'uc' in new_indices[-1].lower():
        new_indices[-1]=new_indices[-1].upper()

# create labels for nodes and node counts
new_labels=[]
for i in range(len(og_counts)):
    node_name=new_indices[i]        # node_name: node_count (node_percentage)
    node_count=og_counts[list(og_counts.keys())[i]]
    node_percentage=str(round(node_count/len(csv_relevant)*100, 2))+'%'
    new_labels.append(node_name + ': ' + str(node_count) + ' \n(' + node_percentage + ')')

# create color maps - these are what will define the colors of the nodes. {node_color: link_color}
node_link_color_maps = {'dodgerblue': '#AED3FF', 'deepskyblue': '#B0E6FF', 'cornflowerblue': '#C1D5F8',
                        'greenyellow': '#DFFFB7', 'lawngreen': '#B8FED7', 'mediumspringgreen': '#CEFEB0', 'darkturquoise': '#B2ECED',
                        'plum': '#F2D9F2', 'pink': '#FFE6EA', 'palevioletred': '#F2C6D4', 'thistle': '#F0E6F0', 'hotpink': '#FFC5E1',
                        'orangered': '', 'salmon': '', 'tomato': '', 'lightcoral': ''}

# extract respective link colors for each of the mappings
link_colors = []
for i in source_indices:
    color_index = list(node_link_color_maps.keys())[i]
    link_colors.append(node_link_color_maps[color_index])

# create the definition of a node
node = dict(pad=15, thickness=15, label=new_labels,
            color=list(node_link_color_maps.keys()))

# create the definition of a link (mapping from one node to another)
link = dict(source=source_indices, target=target_indices,
            value=values, color=link_colors)

# create Sankey diagram
data = go.Sankey(node=node, link=link)

# draw and show the diagram
fig = go.Figure(data)
fig.update_layout(title_text="ULTRA: Human Study Evaluation")
fig.show()

# save the diagram
save_dir = "../../../../../../Downloads/"
fig.write_html(save_dir+"ultra-sankey-human-study-pilot.html")
# fig.write_image(save_dir+"ultra_sankey_human_study.png")

