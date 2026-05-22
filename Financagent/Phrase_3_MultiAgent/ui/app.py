import streamlit as st
st.set_page_config(
    page_title = 'Financagent',
    page_icon= "🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

import uuid
import os
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from Financagent.Phrase_3_MultiAgent.main import run

def init_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'agent_traces' not in st.session_state:
        st.session_state.agent_traces = []
    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

init_session_state()

@st.cache_data
def load_dataset_info():
    data_path = Path(__file__).parent.parent.parent / "data/processed/categorized_data.csv"
    df = pd.read_csv(data_path)
    return {
        'total_transactions': len(df),
        'date_min': df['DATE'].min(),
        'date_max': df['DATE'].max(),
        'categories': df['CATEGORY'].nunique(),
        'total_spending': abs(df[df['IS_DEBIT'] == True]['AMOUNT'].sum())
    }

# -------------SideBar------------------
# Section 1 -> Conversation Button
if st.sidebar.button('➕ New Conversation', use_container_width=True):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.agent_traces = []
    st.rerun()

# Section 2. -> Thread ID
st.sidebar.markdown("### 🧵 Thread ID")
st.sidebar.code(st.session_state.thread_id[:8]+ "...")

# Section 3. -> Dataset Info
info = load_dataset_info()
st.sidebar.markdown("### 📁 Dataset Info")
st.sidebar.metric("Transactions", info["total_transactions"])
st.sidebar.metric("Total Spending", info["total_spending"])
st.sidebar.caption(f"📅 {info['date_min']} → {info['date_max']}")

# Section 4. -> Agent Trace
st.sidebar.markdown("### ⚙️ Settings")
st.session_state.show_traces = st.sidebar.toggle(
    "Show agent reasoning path",
    value = st.session_state.show_traces
)

# ------------ Page Routing ------------
st.sidebar.markdown("---")
st.sidebar.caption(f"💬 {len(st.session_state.messages) // 2} messages")
page = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "📊 Dashboard", "📄 Annual Report"],
    label_visibility="collapsed"
)

if page == "💬 Chat":
    print('Chat')
    from chat import render_chat
    render_chat()
elif page == "📊 Dashboard":
    print('Dashboard')
    from dashboard import render_dashboard
    render_dashboard()
elif page == "📄 Annual Report":
    print('Annual Report')
    from report import render_report
    render_report()

# st.title("💰 Personal Finance Agent")
# st.caption("Multi-agent AI system powered by LangGraph")
st.divider()

if __name__ == "__main__":
    pass