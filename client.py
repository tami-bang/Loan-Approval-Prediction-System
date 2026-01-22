import requests

url = "http://127.0.0.1:8080/predict"

params = {
    "x": 4
}

response = requests.get(url, params=params)

print("status:", response.status_code)
print("response:", response.json())

