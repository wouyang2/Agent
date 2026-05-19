import sys
from pathlib import Path
curr_dir = Path.cwd()
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))
from Financagent.Phrase_3_MultiAgent.agents import analyst, anomaly, search, orchestrator, report_writer, critic
from Financagent.Phrase_3_MultiAgent.core.state import FinanceSystemState
from Financagent.Phrase_3_MultiAgent.core import memory_manager

from langgraph.graph.state import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def retry_node(state: FinanceSystemState):
    """Clears stake agents outputs before retry"""

    cleared = {k:v for k,v in state.agent_outputs.items()
               if k not in state.revision_feedback}
    return {'agent_output': cleared}

memory_saver = MemorySaver()

from IPython.display import Image, display

graph = StateGraph(FinanceSystemState)

graph.add_node('analyst', analyst.analyst_node)
graph.add_node('anomaly', anomaly.anomaly_node)
graph.add_node('search', search.search_node)
graph.add_node('orchestrator_routing', orchestrator.orchestrator_routing_node)
graph.add_node('report_writer', report_writer.report_writer_node)
graph.add_node('orchestrator_synthesis', orchestrator.orchestrator_synthesis_node)
graph.add_node('memory_manager', memory_manager.memory_manager_node)
graph.add_node('critic', critic.critic_node)
graph.add_node('retry', retry_node)

graph.add_edge(START, 'memory_manager')
graph.add_edge('memory_manager', 'orchestrator_routing')

graph.add_conditional_edges('orchestrator_routing', orchestrator.route_to_agents, ['analyst', 'search', 'anomaly'])

# After agents → critic
graph.add_edge('analyst', 'critic')
graph.add_edge('anomaly', 'critic')
graph.add_edge('search', 'critic')

# Critic → conditional routing
graph.add_conditional_edges('critic', orchestrator.route_after_critic, {'retry': 'retry', 'report_writer': 'report_writer', 'orchestrator_synthesis': 'orchestrator_synthesis'})

# Retry → back to agents
graph.add_conditional_edges('retry', orchestrator.route_to_agents,
    ['analyst', 'anomaly', 'search'])


graph.add_edge('orchestrator_synthesis', END)
graph.add_edge('report_writer', END)

app = graph.compile(checkpointer=memory_saver)

# graph_image = app.get_graph().draw_mermaid_png()
#
#
# with open("graph.png", "wb") as f:
#     f.write(graph_image)
#
# print("Graph saved successfully as 'graph.png'!")
