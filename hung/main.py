import requests

API_URL = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Today is a sunny day and I will get some ice cream.",
})

print(output)