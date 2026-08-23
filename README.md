# sfematch: Subject-Field-Expert matching

This is an expert matching project.

## Workflow

### 1. Data collection

The most available data is arXiv.
So, it is good to use it as a source for training.

#### 1.1. Taxonomy collection

Arxiv taxonomies already exist in this project:

sfematch.prepare.arxiv_field.TAX

TAX in:
- CS_FIELDS: Computer Science
- MATH_FIELDS: Mathematics
- STAT_FIELDS: Statistics
- QBIO_FIELDS: Quantitative Biology
- QFIN_FIELDS: Quantitative Finance
- EESS_FIELDS: Electrical Engineering and Systems Science
- PHYSICS_FIELDS: Physics

#### 1.2. Scientific publication collection

ArXiv dataset can be freely downloaded from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
Then, to extract just a subset of papers using a taxonomy:

```sh
python exec/extract_arxiv_domain.py --src arxiv-metadata-oai-snapshot.json --dom cs

```

Replace "cs" with "math", "stat", "bio", "fin", "eng", or "phys".
It will generate a file "cs_papers.jsonl".
Then transform this file into a tsv file using this script:

```sh
python exec/jsonl2tsv.py --src cs_papers.jsonl --dom cs
```



#### 1.3. Expert data collection

This is the most manual data collection step

Start by compiling a list of experts in a text file (each in a line: firstname; family name) like in [](./examples/info/expert_list.txt).
Execute this command to get a file like this [](./examples/info/experts_info_choices.json)

```sh
python exec/collect_infos.py examples/info/expert_list.txt  --out examples/info/experts_info_choices.json
```

Then manually choose the right IDs and compile a file similar to [](./examples/info/experts_info.json).

Also, compile a file for interests manually (cannot be done automatically since most websites prohibit scrapping), like this example [](./examples/info/experts_interests.txt)

### 2. Representation modeling

#### 2.1. Field taxonomy preparation

#### 2.2. Subject modeling

#### 2.3. Expert modeling


### 3.Matching

### 3.2. Subjects encoding

### 3.1. Expert scoring
