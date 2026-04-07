from pyexpat import features

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def wiki_tool():
    wiki_search_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wiki_search_tool