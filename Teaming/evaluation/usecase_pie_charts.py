import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""Create use case pie charts for responses derived from the human study."""

# file paths for use case responses extracted from the human study survey
path_relevant_responses = "../data/human_study_responses/relevant_responses.csv"

# import csv
csv_relevant = pd.read_csv(path_relevant_responses, header=None, names=(
    ['relevancy_scale', 'method', 'usecase', 'comment']))

# check the frequency of each use case
usecase_freq = {}
for i in list(csv_relevant['usecase']):
    if i not in usecase_freq:
        usecase_freq[i] = 1
    else:
        usecase_freq[i] += 1

print(usecase_freq)

# text inside piechart
text = []

for i in usecase_freq:  # for each usecase
    percentage = (usecase_freq[i]/sum(list(usecase_freq.values())))*100
    text.append(
        i+': ' + str(usecase_freq[i]) + ' participants ('+str(round(percentage, 2))+'%)')

# draw and show the diagram
fig = go.Figure(data=[go.Pie(labels=list(usecase_freq.keys()), values=list(
    usecase_freq.values()), texttemplate=text, sort=False)])
fig.update_traces(marker=dict(colors=['maroon', 'brown', 'salmon']))
fig.update_traces(textinfo='value')
fig.show()

# save the diagram
save_dir = "images/"
fig.write_html(save_dir+"piechart_usecases.html")
# fig.write_image(save_dir+"piechart_usecases.png")
