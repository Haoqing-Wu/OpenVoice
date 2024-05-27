import requests
import json
import base64

audio_url = 'https://github.com/myshell-ai/OpenVoice/raw/main/resources/example_reference.mp3'


url = 'http://127.0.0.1:5001/predictions'

# Prepare the payload
payload = json.dumps({
  "input": {
    "ref_audio": audio_url,
    "text": "Did you ever hear a folk tale about a giant turtle?",
    "language": "EN",
    "speaker": "EN-US",
    "speed": "1.0"
    }
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
result = response.json()
content = result.get('output')
header, content = content.split("base64,", 1)
content = base64.b64decode(content)
with open("./out.wav", "wb") as f:
    f.write(content)