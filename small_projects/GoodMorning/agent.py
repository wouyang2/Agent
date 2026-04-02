from tools.weather import get_weather
from tools.news import fetch_news
from tools.calender import fetch_calendar_event
from tools.emails import sendmail

import aisuite as ai
import dotenv
import requests
import os

dotenv.load_dotenv()

CLIENT = ai.Client()

weather_info = get_weather()
news_info = fetch_news()
calender_evnet = fetch_calendar_event()

system_prompt = f"""You are a professional writer that could summarize the information I give you and rewrite them out as briefing in a clean readable format. 
                    And also you are thoughtful assistant that could tailor tone /focus based on the calender."""

user_prompt = f""""Please analyze the given weather info and news info, rewrite and rearrange them in to a brief, clean, readable and comfortable format. 
                    For News Section, please provide the link to the news page if possible.

                Current Weather in My Location: {weather_info}
                Everyday News of Nvidia: {news_info}

                Given the tools and access to my calender, can you please tailor the focus of my day.
                Event : {calender_evnet}

                Return sum up everything you have and return them in a nice, clear format. Thanks.
"""

response = CLIENT.chat.completions.create(
    model='openai:o4-mini',
    messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}],
    temperature=1
)

briefing = response.choices[0].message.content

sendmail(briefing)
