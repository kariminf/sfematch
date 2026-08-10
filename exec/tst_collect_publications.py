import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.collect.collect_publications import extract_dblp_links, extract_researchgate, extract_gscholar


# print(extract_dblp_links("126/6793"))


# print(extract_researchgate("Abdelkrime-Aries"))

print(extract_gscholar("FYJlQL4AAAAJ"))

