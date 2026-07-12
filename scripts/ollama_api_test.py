import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:3b",
    "prompt": "请解释什么是 QLoRA，并说明它为什么省显存。",
    "stream": False
}

response = requests.post(url, json=payload)
response.raise_for_status()

data = response.json()
print(data["response"])
