import dotenv
import requests
import os

dotenv.load_dotenv()

Weather_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

def get_weather(lat = 35.1460249,
                lon = -90.0517638,
                api_key = Weather_API_KEY):

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        print('something went wrong')
        raise e




# if __name__ == '__main__':
#     print(get_weather())