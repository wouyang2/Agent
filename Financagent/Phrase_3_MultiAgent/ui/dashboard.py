import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path

from nltk.ccg import chart


@st.cache_data
def load_data():

    curr_dir = Path(__file__).parent.parent.parent
    data_path = curr_dir / "data/processed/categorized_data.csv"

    df = pd.read_csv(data_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['AMOUNT'] = df['AMOUNT'].abs()
    return df[df['IS_DEBIT'] == True]

def render_dashboard():

    df = load_data()

    st.header('📊 Spending Dashboard')
    col1, col2 = st.columns(2)

    with col1:
        years = sorted(df['YEAR'].unique(), reverse=True)
        selected_year = st.selectbox("Year", years)

    with col2:
        months_in_year = sorted(df[df['YEAR'] == selected_year]['MONTH'].unique())
        month_options = ['All Months'] + months_in_year
        selected_month = st.selectbox("Month", month_options)

    filtered = df[df['YEAR'] == selected_year]
    if selected_month != 'All Months':
        filtered = filtered[filtered['MONTH'] == selected_month]


    m1, m2, m3, m4 = st.columns(4)

    m1.metric("💰 Total Spending", f"${filtered['AMOUNT'].sum():,.2f}")
    m2.metric("📝 Transactions", len(filtered))
    m3.metric("💳 Largest Transaction", f"${filtered['AMOUNT'].max():,.2f}")
    m4.metric("📦 Top Category", filtered.groupby('CATEGORY')['AMOUNT'].sum().idxmax())

    # Bar Chart, Bar Graph by Category
    st.divider()
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Spending by Category")
        category_data = filtered.groupby('CATEGORY')['AMOUNT'].sum().reset_index()
        category_data = category_data.sort_values('AMOUNT', ascending=True)

        fig = px.bar(
            category_data,
            x = 'AMOUNT',
            y = 'CATEGORY',
            orientation = 'h',
            color = 'AMOUNT',
            color_continuous_scale = 'Blues',
            labels = {'AMOUNT': 'Amount ($)', 'CATEGORY': 'Category'}
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Category Distribution
    with chart_col2:
        st.subheader("Spending Distribution")
        fig = px.pie(
            category_data,
            values='AMOUNT',
            names='CATEGORY',
            hole=0.4,  # donut chart
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)


    # Monthly Trend Line Chart
    st.divider()
    st.subheader("Monthly Spending Trend")

    monthly = df[df['YEAR'] == selected_year].groupby('MONTH')['AMOUNT'].sum().reset_index()

    fig = px.line(
        monthly,
        x='MONTH',
        y='AMOUNT',
        markers=True,
        labels={'AMOUNT': 'Total Spending ($)', 'MONTH': 'Month'},
    )
    fig.update_traces(line_color='#6366f1', marker_size=8)
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


    # Transaction table
    st.divider()
    st.subheader("Transactions")

    # Search filter
    search = st.text_input("🔍 Search transactions", placeholder="e.g. Costco, Hilton...")
    table_data = filtered.copy()

    if search:
        table_data = table_data[
            table_data['DESCRIPTION'].str.contains(search, case=False, na=False)
        ]

    # Category filter
    categories = ["All"] + sorted(filtered['CATEGORY'].unique().tolist())
    selected_cat = st.selectbox("Filter by category", categories)
    if selected_cat != "All":
        table_data = table_data[table_data['CATEGORY'] == selected_cat]

    # Display table
    st.dataframe(
        table_data[['DATE', 'DESCRIPTION', 'AMOUNT', 'CATEGORY', 'MONTH']].sort_values('DATE', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            'AMOUNT': st.column_config.NumberColumn('Amount', format='$%.2f'),
            'DATE': st.column_config.DateColumn('Date'),
        }
    )
