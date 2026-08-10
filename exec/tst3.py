import requests
import json
import time

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.collect.openalex import get_works

IDS = ["A5000394355",  "A5036148374",  "A5050441610",  "A5075990520",  "A5090432216",
       "A5003634611",  "A5038137848",  "A5058467315",  "A5081341724",  "A5111861843",
       "A5026630350",  "A5038478629",  "A5058558466",  "A5084368465",  "A5113701843",
       "A5028317399",  "A5044704994",  "A5074641711",  "A5088219626"
]

# for author_id in IDS:
#
#     print(f"Processing {author_id} ...")
#     get_works(author_id, url=f"../JSON2/OpenAlex/{author_id}.json")
#     time.sleep(0.5)

# def get_works(author_id, url="../JSON2/OpenAlex/"):
#     works = requests.get(
#         f"https://api.openalex.org/works?filter=author.id:{author_id}"
#     ).json()
#
#     outf = url + author_id + ".json"
#
#     with open(outf, "w", encoding="utf8") as f:
#         json.dump(
#             works,
#             f,
#             indent=4,
#             ensure_ascii=False
#         )
#
# filename = "../JSON2/experts_info.json"
#
# with open(filename, encoding="utf8") as f:
#     experts = json.load(f)
#
#
#
# for eid in experts:
#
#     print(f"Processing {eid} ...")
#     expert = experts[eid]
#     author_id = expert["profiles"]["openalex"]
#     if author_id:
#         get_works(author_id)
#     time.sleep(0.5)

author_id = "A5060914734"
print(f"Processing {author_id} ...")
get_works(author_id, url=f"../{author_id}.json")
