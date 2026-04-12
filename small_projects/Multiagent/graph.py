from langgraph.graph import StateGraph, END, START
from typing_extensions import Literal

from state import State
from agents.searcher import search_agent
from agents.analyst import analyze_sources
from agents.writer import writer
from agents.critic import critic

def router(state: State) -> Literal["end", 'search_agent']:

    if state['revision_count'] >= 4:
        return 'end'

    critic_message = state['critique']

    if not critic_message:
        return 'end'

    if any(word in critic_message[-1].content.lower() for word in ['insufficient', 'missing']):
        return 'search_agent'

    return 'end'


graph = StateGraph(state_schema=State)

graph.add_node('search_agent',search_agent)
graph.add_node('analyze_sources',analyze_sources)
graph.add_node('writer',writer)
graph.add_node('critic',critic)

graph.add_edge(START, 'search_agent')
graph.add_edge('search_agent', 'analyze_sources')
graph.add_edge('analyze_sources', 'writer')
graph.add_edge('writer', 'critic')
graph.add_edge('critic', END)

graph.add_conditional_edges('critic',
                            router,
                            {
                                'search_agent': 'search_agent',
                                'end' : END
                            })

workflow = graph.compile()



# Visualizing
graph_image = workflow.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_image)

