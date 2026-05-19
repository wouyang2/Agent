import pandas as pd
from langchain_openai import ChatOpenAI
import dotenv
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from itertools import batched
from pydantic import BaseModel, Field
from typing import List

dotenv.load_dotenv()

class CategorizedTransaction(BaseModel):
    description: str = Field(description="The original transaction description")
    label: str = Field(description="The category label generated for this description")

class TransactionCategory(BaseModel):
    items: List[CategorizedTransaction]

CATEGORIES = ["Food & Dining", "Groceries", "Gas", "Subscription", "Entertainment", "Technology", "Alcohol", "Smoke", "Payment", "Shopping", "Transportation", "Shipping", "Charity"]

llm = ChatOpenAI(model = 'gpt-4o-mini')

def process_data(data) -> pd.DataFrame:

    # Parse Date into proper datetime
    data['DATE'] = pd.to_datetime(data['DATE'])

    # Add 'is_debit' Column
    data['IS_DEBIT'] = data['AMOUNT'] < 0

    # Add 'month' column
    data['MONTH'] = data["DATE"].dt.to_period("M")

    # Add 'year' column
    data['YEAR'] = data['DATE'].dt.to_period("Y")

    data = data.drop(columns=['CHECK #'])

    return data

def categorize_batch(descriptions, model):

    categories_str = "\n".join(f"- {c}" for c in CATEGORIES)
    descriptions_number_list = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))

    system_prompt = SystemMessage("You are a financial transaction categorization engine. You only return valid Python lists.")

    user_prompt = HumanMessage(f"""Categorize each of the {len(descriptions)} transaction descriptions below.

                    VALID CATEGORIES (choose exactly one per description):
                    {categories_str}
                
                    RULES:
                    - Return a JSON object where each key is the exact description text and each value is the category label.
                    - Every label must exactly match one from the valid categories above
                    - No compound labels, no slashes, no extra text, no markdown
                    - If unsure, use "Other"
                
                    EXAMPLE (for 2 descriptions):
                    ["TEXAS ROAD HOUSE" : "Food & Dining", "KROGER": "Groceries"]
                
                    DESCRIPTIONS:
                    {descriptions_number_list}
    """)

    messages = [system_prompt, user_prompt]


    output = model.invoke(messages)

    # Extracting labels from the output

    categories = []

    for description, label in output.items:
        categories.append(label[1])

    # print("Categories list: ", categories)
    # print("Categories length: ", len(categories))

    return categories

def auto_categorize(data, model):

    structured_model = model.with_structured_output(TransactionCategory, method = "function_calling")

    descriptions = list(data['DESCRIPTION'])

    final_category = []

    for chunk in batched(descriptions, 20):
        category = categorize_batch(chunk, structured_model)
        final_category.extend(category)

    data['CATEGORY'] = final_category

def run_ingestion(input_path, output_path):

    with open(input_path) as file:
        df = pd.read_csv(file)

    data = process_data(df)

    # Turn description into list
    auto_categorize(data, llm)

    # Write processed df to csv
    data.to_csv(output_path, index=False)

    print("Data Shape: ",data.shape)
    print(data.head())


if __name__ == '__main__':
    run_ingestion("/Users/kiwi/Developer/Agent/Financagent/data/raw/CreditCard.csv", "/Users/kiwi/Developer/Agent/Financagent/data/processed/categorized_data.csv")

