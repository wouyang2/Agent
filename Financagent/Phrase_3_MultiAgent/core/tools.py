import os
import dotenv
import pandas as pd
import json
import sys
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain_chroma import Chroma


dotenv.load_dotenv()
curr_dir = Path(__file__).parent.parent
base_dir = curr_dir.parent.parent
sys.path.insert(0, str(base_dir))
from Financagent.rag import query_transactions

# Load the embedding model
embed = OpenAIEmbeddings(model = 'text-embedding-3-small')

# Load the normalized data file
data_path = os.path.join(base_dir, 'Financagent/data/processed/categorized_data.csv')
with open(data_path) as csvfile:
    df = pd.read_csv(csvfile)

# Vector Store
vs_path = os.path.join(curr_dir.parent, 'chroma_db')
vs = Chroma(collection_name='transactions', embedding_function=embed, persist_directory=vs_path)

"""---------------------------- Helpers ----------------------------"""

def helper(dataframe, mean_df, std_df):

    is_outlier = []
    distance = []

    for i in range(len(dataframe)):
        cate = dataframe.iloc[i]['CATEGORY']

        for j in range(len(mean_df)):
            if cate == mean_df.iloc[j]['CATEGORY']:
                mean = mean_df.iloc[j]['AMOUNT']
                std = std_df.iloc[j]['AMOUNT']

                is_outlier.append(abs(dataframe.iloc[i]["AMOUNT"]) > abs(mean) + 2*std)
                distance.append(abs(dataframe.iloc[i]["AMOUNT"]) - abs(mean))

    dataframe['is_outlier'] = is_outlier
    dataframe['distance'] = distance

    return dataframe

def detect_anomaly_month_helper(year) -> list[str]:
    # Calculate the mean of the transaction of each category
    mean = df.groupby('CATEGORY', as_index=False).agg({'AMOUNT': 'mean'})

    # Calculate the standard deviation
    std = df.groupby('CATEGORY', as_index=False).agg({'AMOUNT': 'std'})

    year_months = df[df['YEAR'] == int(year)]['MONTH'].unique()

    anomaly_months = []
    for month in year_months:
        month_df = df[df['MONTH'] == month].copy()
        month_df = helper(month_df, mean, std)
        if month_df['is_outlier'].any():
            anomaly_months.append(month)

    return sorted(anomaly_months)


"""------------------ Tools ------------------"""
@tool
def get_monthly_spending_summary(month=None, category = None, year = None):

    """
    Returns a spending summary for a given month.
    Use this when the user asks about total spending over a year, monthly expenses,
    or a breakdown of spending by category.

    Args:
        month: Month in YYYY-MM format, e.g. '2024-03'.
        category: Optional. A spending category e.g. 'Food & Dining', 'Groceries',
                  'Transportation'. If omitted, returns a full breakdown across
                  all categories.
        year: Optional. Filter results to a specific year in YYYY format.
    Returns:
        A formatted string with total spending, transaction count, and
        optionally a per-category breakdown.
    """
    # Debugging
    # print(f"[TOOL DEBUG] month={month}, year={year}, category={category}")
    # print(f"[TOOL DEBUG] df shape: {df.shape}")
    # print(f"[TOOL DEBUG] df YEAR unique values: {df['YEAR'].unique()}")
    # print(f"[TOOL DEBUG] df YEAR dtype: {df['YEAR'].dtype}")

    if not month and not year:
        return "Please provide either a month (YYYY-MM) or a year (YYYY) to get a spending summary."

    if year:
        df_filtered = df[df['YEAR'] == int(year)]
        df_filtered = df_filtered[df_filtered['IS_DEBIT'] == True]

        if category:
            df_filtered = df_filtered[df_filtered['CATEGORY'] == category]
            total_spending = df_filtered['AMOUNT'].sum().round(2)
            monthly_spending = df_filtered.groupby('MONTH')['AMOUNT'].sum().round(2)
            categorized_spending = df_filtered.groupby('CATEGORY')['AMOUNT'].sum().round(2)
            summary = f"In {year}, the total spending is {total_spending}.\n\nThe Spending of each month is \n{monthly_spending}.\n\nThe breakdown of each categorized spending is\n{categorized_spending}"

        else:
            categorized_breakdown = df_filtered.groupby(['CATEGORY'], as_index=False)['AMOUNT'].sum().round(2)
            categorized_breakdown.sort_values(by=['AMOUNT'], ascending=False, inplace=True)
            total_spending = df_filtered['AMOUNT'].sum().round(2)
            monthly_spending = df_filtered.groupby('MONTH')['AMOUNT'].sum().round(2)
            summary = f"In {year}, the total spending is {total_spending}. \nThe Spending of each month is \n{monthly_spending}. \n Categorized Breakdown is: \n{categorized_breakdown}."

        return summary


    df_filtered = df[df['MONTH'] == month]
    df_filtered = df_filtered[df_filtered['IS_DEBIT'] == True]

    if category:
        df_filtered = df_filtered[df_filtered['CATEGORY'] == category]
        num_transactions = len(df_filtered)
        total_spending = df_filtered['AMOUNT'].sum().round(2)
        summary = f"In {month}, the total spending on {category} is {total_spending}. And there are {num_transactions} transactions in total."

    else:
        num_transactions = len(df_filtered)
        total_spending = df_filtered['AMOUNT'].sum().round(2)
        categorized_spending = df_filtered.groupby(['CATEGORY'], as_index=False)['AMOUNT'].sum().round(2)
        categorized_breakdown = '\n'.join([f"Category: {row.CATEGORY}  Amount: {row.AMOUNT}" for row in categorized_spending.itertuples()])
        summary = f"""In {month}, the total spending is {total_spending} over {num_transactions} transactions.\n\nThe breakdown of each category is \n{categorized_breakdown}."""



    return summary

@tool
def search_transactions(question, month = None, category= None, year = None):

    """
    Semantically searches transaction history using a natural language question.
    Use this when the user asks about specific vendors, unusual charges, or
    wants to find transactions matching a description rather than an exact category.

    Args:
        question: A natural language question e.g. 'any charges at Amazon last month?'
        month: Optional. Filter results to a specific month in YYYY-MM format.
        category: Optional. Filter results to a specific category e.g. 'Shopping'.
        year: Optional. Filter results to a specific year in YYYY format.
    Returns:
        A formatted string listing the most relevant matching transactions
        with their metadata and descriptions.
    """

    filters = {'is_debit': 'true'}

    if category:
        filters['category'] = category

    if month:
        filters['month'] = month

    if year:
        filters['year'] = year

    # results = query_transactions(question,  collection= vs._collection, filters = filters)
    results = query_transactions(question, collection=vs, filters=filters)

    # Format the result into an agent readable string
    outputs = '\n'.join([f"page_content: {doc.page_content} metadata: {doc.metadata}" for doc in results])

    # Debugging
    # print(f"[SEARCH DEBUG] question={question}, filters={filters}")
    # print(f"[SEARCH DEBUG] results count: {len(results)}")
    # print(f"[SEARCH DEBUG] results: {results[0]}")

    return outputs


@tool
def detect_anomalies(month):

    """
    Detects unusually large transactions in a given month by comparing each
    transaction against the historical average for its category.
    Use this when the user asks about suspicious charges, anything unusual,
    or unexpected spending spikes.

    Args:
        month: Month in YYYY-MM format, e.g. '2024-03'.
    Returns:
        A formatted string listing transactions that are more than 2 standard
        deviations above their category's historical average, including the
        amount and how far above normal each transaction is.
    """

    # Calculate the mean of the transaction of each category
    mean = df.groupby('CATEGORY', as_index=False).agg({'AMOUNT':'mean'})

    # Calculate the standard deviation
    std = df.groupby('CATEGORY', as_index=False).agg({'AMOUNT':'std'})

    # Flag the transaction of given month that are more than 2 standard deviations above that category's average
    df_filtered = df[df['MONTH'] == month]

    df_filtered = helper(df_filtered, mean, std)

    # Return the anomalies with the amount and how far they above the mean as a list
    ret = []
    for row in df_filtered.itertuples():
        if row.is_outlier:
            ret.append({'transactions': row.AMOUNT, 'distance': row.distance, 'description': row.DESCRIPTION})


    s = json.dumps(ret)

    return s

@tool
def get_monthly_comparison(category = None, year = None):

    """
    Compares spending month over month across the entire transaction history.
    Use this when the user asks about spending trends, whether they are spending
    more or less overtime, or how a specific category has changed across months.

    Args:
        category: Optional. A spending category e.g. 'Entertainment'. If omitted,
                  compares total spending across all categories month over month.
        year: Optional. Filter results to a specific year in YYYY format.
    Returns:
        A formatted string showing total spending per month in chronological order,
        optionally filtered to a single category.
    """

    # Group df by month and optionally category
    if category:
        df_filtered = df[df['CATEGORY'] == category]
        spending_per_month = df_filtered[df_filtered['IS_DEBIT'] == True].groupby(['MONTH'], as_index=False)['AMOUNT'].sum()
    else :
        spending_per_month = df[df["IS_DEBIT"] == True].groupby(['MONTH'], as_index=False)['AMOUNT'].sum()

    if year:
        spending_per_month = spending_per_month[spending_per_month['MONTH'].str.startswith(str(year))]

    # Return summary showing the spending per month
    """ 
        Summary in the following format: [{'month': 'amount'}, {'month': 'amount'}]
    """

    ret = []
    for row in spending_per_month.itertuples():
        info = {
            row.MONTH: row.AMOUNT,
        }
        ret.append(info)


    s = json.dumps(ret)

    return s

@tool
def get_full_annual_report (year: int):
    """
    Generates a comprehensive financial data dump for a full year.
    Use this when the user explicitly asks for a detailed report, full analysis,
    or complete summary of a year's finances.

    Args:
        year: The year to generate the report for, as an integer e.g. 2025.

    Returns:
        A structured string containing:
        - Total spending for the year
        - Category breakdown aggregated across all months
        - Month-by-month spending totals
        - Top 5 largest individual transactions
        - List of months containing anomalous transactions
    """


    df_filtered = df[df['YEAR'] == int(year)]
    df_filtered = df_filtered[df_filtered['IS_DEBIT'] == True]

    total_spending = abs(df_filtered['AMOUNT'].sum())
    categorized_breakdown = df_filtered.groupby('CATEGORY', as_index=False)['AMOUNT'].sum()
    month_breakdown = df_filtered.groupby('MONTH', as_index=False)['AMOUNT'].sum()
    five_largest_transaction = df_filtered.nsmallest(5, ['AMOUNT']) # returning the rows

    anomaly_months = detect_anomaly_month_helper(year)

    # Debugging
    # print('total_spending: ', total_spending)
    # print('categorized_breakdown: ', categorized_breakdown)
    # print('month_breakdown: ', month_breakdown)
    # print('five_largest_transaction: ', five_largest_transaction)
    # print('Any months with anomalies: ', anomaly_months)

    return f"""
            ANNUAL REPORT FOR {year}
        
            Total Spending: ${abs(total_spending):.2f}
            
            Category Breakdown:
            {categorized_breakdown.to_string(index=False)}
            
            Month-by-Month:
            {month_breakdown.to_string(index=False)}
            
            Top 5 Largest Transactions:
            {five_largest_transaction[['DATE', 'DESCRIPTION', 'AMOUNT', 'CATEGORY']].to_string(index=False)}
            
            Months with Anomalies: {', '.join(anomaly_months) if anomaly_months else 'None detected'}
            """



if __name__ == '__main__':
    vs_path = os.path.join(curr_dir.parent, 'chroma_db')
    print(f"[PATH DEBUG]: CURRENT DIR: {curr_dir}")
    print(f"[PATH DEBUG]: BASE DIR: {base_dir}")
    print(f"[PATH DEBUG] ChromaDB path: {vs_path}")
    print(f"[PATH DEBUG] Path exists: {os.path.exists(vs_path)}")





