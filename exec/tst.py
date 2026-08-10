import requests
import json

author = requests.get(
    "https://api.openalex.org/authors?search=Abdelkrime Aries"
).json()

author_id = author["results"][0]["id"]

works = requests.get(
    f"https://api.openalex.org/works?filter=author.id:{author_id}"
).json()


with open("outfile.json", "w", encoding="utf8") as f:
        json.dump(
            works,
            f,
            indent=4,
            ensure_ascii=False
        )
