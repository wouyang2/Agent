import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from Financagent.Phrase_3_MultiAgent.main import stream_agent
import uuid
import pandas as pd

@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / 'data/processed/categorized_data.csv'

    df = pd.read_csv(data_path)

    df_filtered = sorted(df['YEAR'].unique(), reverse=True)

    return df_filtered


def render_report():
    st.header("📄 Annual Financial Report")
    st.caption("Generate a comprehensive AI-powered analysis of your yearly finances.")

    years = load_data()

    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox('Select Year', years)

    with col2:
        st.write("")
        generate = st.button('🚀 Generate Report', width='stretch')

    # Generate Report and Display
    if generate:
        with st.spinner(f"Generating report for {selected_year}..., this may take a few minutes."):
            result = run(f"Generate a full annual report for {selected_year}", thread_id=str(uuid.uuid4()))

        st.divider()
        st.markdown(result['response'])

        # Download button
        st.divider()
        st.download_button(
            label="⬇️ Download Report as Text",
            data=result['response'],
            file_name=f"finance_report_{selected_year}.md",
            mime="text/markdown"
        )

    else:
        st.info("👆 Select a year and click Generate Report to create your annual financial analysis.")

        # Preview of what the report includes
        st.markdown("""
        **The report includes:**
        - 📋 Executive Summary
        - 🗂 Spending by Category  
        - 📈 Monthly Trends
        - 💳 Notable Transactions
        - ⚠️ Anomalies & Unusual Activity
        - 💡 Actionable Observations
        """)