from langchain_tavily import TavilySearch

import dotenv
dotenv.load_dotenv()

def tavily_tool():
    tavily_search_tool = TavilySearch(include_images = True, time_range= 'week')

    return tavily_search_tool



