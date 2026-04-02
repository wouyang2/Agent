import requests
import os
from datetime import datetime, timedelta, date
import dotenv

dotenv.load_dotenv()

def fetch_news(start_date = date.today() - timedelta(days=1),
               end_date = date.today() - timedelta(days=1),
               topic = 'Nvidia',
               language = 'en',
               api_key = os.environ.get('NEWS_API_KEY')):

    url = f"https://newsapi.org/v2/everything?q={topic}&from={start_date}&to={end_date}&language={language}&sortBy=popularity&apiKey={api_key}"
    url.format(start_date, end_date, topic, api_key)

    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json()
        return articles
    except requests.exceptions.RequestException as e:
        print("something went wrong")
        raise e


# if __name__ == "__main__":
#     print(fetch_news())