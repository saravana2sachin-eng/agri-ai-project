import requests

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&timezone=auto"

    response = requests.get(url)
    data = response.json()

    # Temperature
    temperature = data['current_weather']['temperature']

    # Rainfall (daily)
    rainfall = data['daily']['precipitation_sum'][0]

    return temperature, rainfall