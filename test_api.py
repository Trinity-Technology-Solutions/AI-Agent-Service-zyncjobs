import requests

url = "http://localhost:8001/plagiarism/check"
data = {
    "text": "ஸ்ரீநந்திகியோர் கோயங்காவின் இறுதிச் சடங்குகள் அனைத்தும், கடந்த ஜூலை 15ஆம் தேதி அவர் உருவாக்க உதவிய ஹரியானா ஹிசார் மாவட்டத்தில் இருக்கும் அக்ரோஹா தாமில் உள்ள கோயங்கா உத்யான் பகுதியில் நடைபெற்றது. எஸ்செல் குழுமத் தலைவர் டாக்டர் சுபாஷ் சந்திரா இறுதிச் சடங்குகளை மேற்கொண்டார். பிரதமர் மோடி உள்பட பல்வேறு அரசியல் தலைவர்கள், பிரபலங்கள், நட்சத்திரங்கள், அதிகாரிகள், பொதுமக்கள் என பலரும் ஸ்ரீநந்திகியோர் கோயங்கா மறைவிற்கு ஆழ்ந்த இரங்கலை தெரிவித்தனர்."
}

try:
    print("Sending request to API...")
    res = requests.post(url, json=data)
    res.raise_for_status()
    import json
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print("Error:", e)
