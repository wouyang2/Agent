import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Financagent.Phrase_3_MultiAgent.main import run

def render_trace(trace):
    """Renders the agent reasoning path inside the expander. """
    routes_to, critic_score, revision = st.columns(3)

    with routes_to:
        st.markdown("🔀 Routed to")

        # for agent in trace.get('routes', ""):
        #     st.write(agent)
        st.write(trace.get('routed_to'))

    with critic_score:
        st.markdown("🔍 Critic scores")
        st.write(trace.get('critic_scores'))

    with revision:
        st.markdown("🔄 Revisions")
        st.write(trace.get('revision_count'))

    st.markdown("**🛠 Tools called:**")
    for tool in trace.get('tool_history', []):
        tool_name = tool.get('tools') or tool.get('tool', 'unknown')
        tool_args = tool.get('args', {})
        st.code(f"{tool_name}({tool_args})")

def render_message(message, trace, show_traces: bool):

    if message['role'] == 'user':
        with st.chat_message('user'):
            st.markdown(message['content'])

    elif message['role'] == 'assistant':
        with st.chat_message('assistant'):
            st.markdown(message['content'])
            if show_traces and trace:
                with st.expander("🔍 Agent reasoning path"):
                    render_trace(trace)

def render_chat():
    if not st.session_state.messages:
        st.info("👋 Ask me anything about your finances. Try: 'How much did I spend in 2025?'")

    for message in st.session_state.messages:

        trace = message.get('trace')
        render_message(message, trace, st.session_state.show_traces)

    user_input = st.chat_input("Ask about your finance.....")

    if user_input:

        st.session_state.messages.append({'role':'user', 'content':user_input})

        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    result = run(user_input, st.session_state.thread_id)
                except Exception as e:
                    st.error(f"Agent encountered an error: {str(e)}")
                    st.stop()

            st.markdown(result['response'])

            # Showing reasoning path if needed
            if st.session_state.show_traces and result:
                with st.expander("🔍 Agent reasoning path"):
                    render_trace(result)

        # Store assistant message
        st.session_state.messages.append({
            'role': 'assistant',
            'content': result['response'],
            'trace': result
        })