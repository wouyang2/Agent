import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..agent import run_agent
import pytest
import uuid

CATEGORIES = ["Food & Dining", "Groceries", "Gas", "Subscription", "Entertainment", "Technology", "Alcohol", "Smoke", "Payment", "Shopping", "Transportation", "Shipping", "Charity", "Other"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

@pytest.fixture
def thread():
    return str(uuid.uuid4())

def run_conversation(turns: list[str], thread_id: str) -> list[str]:

    responses = []

    for userInput in turns:
        response = run_agent(user_input=userInput, thread_id=thread_id)
        responses.append(response)

    return responses

class TestContextRetention:   # does agent remember context across turns?

    def test_year_followup(self, thread):
        turns = ['How much did I spend in 2024?', 'How about 2025']
        result = run_conversation(turns=turns, thread_id=thread)
        print("Response: ", result)

        assert '2025' in result[1]
        assert result[1].count('2025') > result[1].count('2024')
        assert all(phrase not in result[1].lower() for phrase in ['clarify', 'please specify'])  # ✅

    def test_category_follow_up_after_month(self, thread):
        turns = ['What did I spend in August 2025?', 'Which category was highest?']
        result = run_conversation(turns=turns, thread_id=thread)
        print("\nTurn 1:", result[0])
        print("\nTurn 2:", result[1])
        assert '$' in result[1]
        assert any(item in result[1] for item in CATEGORIES)
        assert all(item not in result[1] for item in ['clarify', 'please specify', 'which month'])

    def test_yes_confirmation(self, thread):
        turns = ['What about March', 'yes']
        result = run_conversation(turns=turns, thread_id=thread)
        print('Test_yes_confirmation:', result)

        assert all(item not in result[1] for item in ['Hello', 'How can I help you?'])
        assert any(items in result[1] for items in CATEGORIES)
        assert 'March' in result[1]

    def test_month_without_year(self, thread):
        turns = ['How much did I spend in 2025', 'How about in September']
        result = run_conversation(turns=turns, thread_id=thread)
        assert any(item in result[1] for item in ['2025-09', 'September 2025'])
        assert all(phrase not in result[1].lower() for phrase in ['which year', 'please specify'])


class TestToolRouting:

    def test_routes_to_spending_summary(self, thread):
        turns = ['How much did I spend in March 2026?']
        result = run_conversation(turns=turns, thread_id=thread)
        assert '2026' in result[0]
        assert '$' in result[0]
        assert any(item in result[0] for item in CATEGORIES)
        assert all(item not in result[0] for item in ['trend', 'month over month'])

    def test_routes_to_anomaly_detection(self, thread):
        turns = ['Anything unusual in August 2025?']
        result = run_conversation(turns=turns, thread_id=thread)
        assert '$' in result[0]
        assert any(item in result[0] for item in ['unusual', 'above average', 'exceeds', 'spike'])
        assert any(char.isalpha() and char.isupper() for char in result[0])

    def test_route_to_trend(self, thread):
        turns = ['Show me my spending trend for 2025']
        result = run_conversation(turns=turns, thread_id=thread)
        assert sum(1 for item in MONTHS if item in result[0]) >= 3
        assert result[0].count('$') >= 2
        assert all(item not in result[0] for item in ['above average', 'anomaly'])

    def test_routes_to_search(self, thread):
        turns = ['Find my Costco charge']
        result = run_conversation(turns=turns, thread_id=thread)
        print(result)
        assert any(item in result[0] for item in ['COSTCO', 'COST', 'Costco'])
        assert '$' in result[0]
        assert 'breakdown' not in result[0].lower()


# Helper
CLARIFICATION_PHRASES = [
    "could you please clarify",
    "could you please specify",
    "which year",
    "which month",
    "please provide",
    "please let me know",
    "please specify",
    "can you clarify",
]

def assert_no_clarification(response: str):
    for phrase in CLARIFICATION_PHRASES:
        assert phrase not in response.lower(), \
            f"Agent asked for clarification unnecessarily: '{phrase}' found in response"

class TestClarificationBehavior:

    def test_no_clarification_on_year_followup(self, thread):
        turns = ['How much did I spend in 2024?', 'How about 2025']
        result = run_conversation(turns=turns, thread_id=thread)

        print("test_no_clarification_on_year_followup: ", result)

        assert_no_clarification(result[1])
        assert '2025' in result[1]

    def test_no_clarification_on_category_followup(self, thread):
        turns = ['What did I spend in August 2025?', 'Which category was highest?']
        result = run_conversation(turns=turns, thread_id=thread)

        assert_no_clarification(result[1])
        assert any(item in result[1] for item in CATEGORIES)

    def test_no_clarification_on_month_followup(self, thread):
        turns = ['How much did I spend in 2026?', 'How about in March?']
        result = run_conversation(turns=turns, thread_id=thread)
        assert_no_clarification(result[1])
        assert any(item in result[1] for item in ["2026-03", "March 2026", "March"])

    def test_no_clarification_on_yes_confirmation(self, thread):
        turns = ['What about September?', 'yes']
        result = run_conversation(turns=turns, thread_id=thread)

        assert all(item not in result[1] for item in ["Hello", "how can I assist"])
        assert 'September' in result[1]


    def test_first_turn_clarification_is_acceptable(self, thread):
        turns = ['Which category did I spend most on?']
        result = run_conversation(turns=turns, thread_id=thread)

        assert len(result[0]) > 0
        asked_clarification = any(item in result[0].lower() for item in ['clarify', 'specify', 'which year', 'which month', 'ask'])
        returned_data = '$' in result[0] or any(item in result[0] for item in CATEGORIES)

        print ('Response: ', result[0])

        assert asked_clarification or returned_data, \
            "Agent neither clarified nor returned useful data"

