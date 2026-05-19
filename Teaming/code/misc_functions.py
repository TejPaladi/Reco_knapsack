# Import/download libraries
import BeautifulSoup
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

"""Misc functions regarding data extraction, export, etc."""

# Function that takes [researcher_interests] and [matched_proposal_info] as input, and returns [common skills b/n researcher interests and matched proposals]


def extract_title_synopsis_from_rfp(proposal_url):
    # print("Proposal URL: "+proposal_url)
    html = urlopen(proposal_url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill script/style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text (webpage content)
    text = soup.get_text()
    # print(text)

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    lines = text.split('\n')

    # Title - first line on the page
    title = lines[0]

    try:
        # Synopsis - search for it, save its index, and extract it
        get_synopsis_index = lines.index("Synopsis of Program:")
        synopsis = lines[get_synopsis_index+1]
        # print("\nSynopsis: "+synopsis)
    except:  # Fallback response (TEMPORARY)
        synopsis = ""

    return title, synopsis
