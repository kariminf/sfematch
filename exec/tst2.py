import requests
import json

import time
import requests

session = requests.Session()
session.headers.update({
    "User-Agent": "sfematch/1.0 (kariminfo0@gmail.com)"
})

def find_author(name):
    response = session.get(
        "https://api.openalex.org/authors",
        params={
            "search": name,
            "mailto": "kariminfo0@gmail.com"
        },
        timeout=10,
    )
    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        return ""

    return results[0]


filename = "../JSON2/experts_info.json"

with open(filename, encoding="utf8") as f:
    experts = json.load(f)


txt = ""

for eid in experts:
    print(f"Processing {eid} ...")
    expert = experts[eid]
    name = " ".join(expert["given_names"]) + " " + expert["family_name"]
    alexid = find_author(name)
    txt += f"{eid}\t{alexid}\n"
    time.sleep(0.5)

with open("openalex.txt", "w", encoding="utf-8") as f:
    f.write(txt)
