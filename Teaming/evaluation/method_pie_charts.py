import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

"""Create method pie charts for responses derived from the human study."""

# file paths for method responses extracted from the human study survey
path_relevant_responses = "../data/human_study_responses/relevant_responses.csv"

# import csv
csv_relevant = pd.read_csv(path_relevant_responses, header=None, names=(
    ['relevancy_scale', 'method', 'usecase', 'comment']))

# check the frequency of each method
method_freq = {}
for i in list(csv_relevant['method']):
    if i not in method_freq:
        method_freq[i] = 1
    else:
        method_freq[i] += 1

print(method_freq)

# text inside piechart
text = []

for i in method_freq:  # for each method
    percentage = (method_freq[i]/sum(list(method_freq.values())))*100
    text.append(
        i+': ' + str(method_freq[i]) + ' responses ('+str(round(percentage, 2))+'%)')

# draw and show the diagram
fig = go.Figure(data=[go.Pie(labels=list(method_freq.keys()), values=list(
    method_freq.values()), texttemplate=text, sort=False)])
fig.update_traces(marker=dict(colors=['maroon', 'brown', 'salmon', 'tomato']))
fig.update_traces(textinfo='value')
fig.show()

# save the diagram
save_dir = "images/"
fig.write_html(save_dir+"piechart_methods.html")
# fig.write_image(save_dir+"piechart_methods.png")