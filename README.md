# sfematch: Subject-Field-Expert matching

This is an expert matching project.

## Workflow

### 1. Data collection and preparation

The most available data is arXiv.
So, it is good to use it as a source for training.

#### 1.1. Taxonomy collection

Arxiv taxonomies already exist in this project: cs, math, stat, bio, fin, phys.
Othor antologies can be afforded if you afford mappings from the original you used.
The following sections will discuss this.

#### 1.2. Scientific publication collection

ArXiv dataset can be freely downloaded from [Kaggle:arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).
Then, to extract just a subset of papers using a taxonomy:

```sh
python exec/extract_arxiv_domain.py --src arxiv-metadata-oai-snapshot.json --dom cs

```

Replace "cs" with "math", "stat", "bio", "fin", "eng", or "phys".
It will generate a file "cs_papers.jsonl".
Then transform this file into a tsv file "cs_papers.tsv" using this script:

```sh
python exec/jsonl2tsv.py --src cs_papers.jsonl --dom cs
```

Split the dataset into train/test with a balanced percentage of labels.
Given a file "cs_papers.tsv", this will generate "cs_papers_train.tsv" and "cs_papers_test.tsv".
Also it will print a pretty table with statistics on fields distribution. 

```sh
python exec/train_test_split.py cs_papers.tsv  --test-size 0.2
```

To separate the text from labels, use the following script.
In this case, the file "cs_papers_train.tsv" will be separated into "cs_papers_train_text.tsv" and "cs_papers_train_labels.tsv".

```sh
python separate_text_labels.py cs_papers_train.tsv
python separate_text_labels.py cs_papers_test.tsv
```

To map the labels from a taxonomy to another use the following script.
By default, it maps from arXiv CS to both ACM CCS 2012 L1 and a one we defined called CCSF.
In case, you are dealing with another taxonomy, such as arXiv physics or else, prepare a mapping file similar to that in [ccsf_mapping.json](./examples/subject/ccsf_mapping.json); it contains a name used to rename the output file, a list of target taxonomy, a mapping function for each label of the source taxonomy a list of indices of the target taxonomy (starting from 0).

```sh
python exec/map_labels.py cs_papers_train_labels.tsv --map ./examples/subject/ccsf_mapping.json
```

For arXiv computer science papers:
- We created a dataset for three taxonomies arXiv, ACM CCS 2012 L1 and our CCSF.
The dataset is distributed on Kaggle under CC-BY 4 [cs-papers-arxiv-multilabel](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-multilabel)
- Also, to save time for researchers, we created another dataset containing only embeddings of the title+abstract using many PLMs: BERT, Sentence-BERT, SciBERT, and SPECTER2. It is distributed under CC-BY 4 [cs-papers-arxiv-embeddings](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-embeddings).


#### 1.3. Expert data collection

This is the most manual data collection step.

Start by compiling a list of experts in a text file (each in a line: firstname; family name) like in [expert_list.txt](./examples/info/expert_list.txt).
Execute this command to get a file like this [experts_info_choices.json](./examples/info/experts_info_choices.json)

```sh
python exec/collect_infos.py examples/info/expert_list.txt  --out examples/info/experts_info_choices.json
```

Then manually choose the right IDs and compile a file similar to [experts_info.json](./examples/info/experts_info.json).

Also, compile a file for interests manually (cannot be done automatically since most websites prohibit scrapping), like this example [experts_interests.txt](./examples/info/experts_interests.txt)

<!-- OpenAlex -->
**Work processing**

OpenAlex is a good source for expert works since it affords an API with a daily quota.
This script downloads openalex files for each author to a folder.
If the script is interepted, it can skip the ones already having a file downloaded.
In this example, our output is a folder called "openalex".
This script, will create a folder called "openalex/profiles" to save each expert's openalex profile.
Then, it extracts individual works to a folder "openalex/works". 
```sh
python exec/collect_works.py examples/info/experts_info.json --out openalex
```

To extract abstract issues use the following script; we specified the main folder as "openalex" and it should contain a folder "openalex/works".
This script will create a file "openalex/abstract_issues_report.json" containing works without abstracts and also redundant works (probably the same work).
Then you can manually process them.
```sh
python exec/find_abstract_issues.py openalex
```
Then, you can index expert works using the script.
These works are processed to extract each author and its works list into "openalex/indexed_works.json".
Finaly, you have to manually copy the works from "openalex/indexed_works.json" to "openalex/expert_works.json".
It is done manually because authors names are not always standard (in the indexed works you can find the same author with many entries)
```sh
python exec/index_expert_works.py openalex
```

Finally to transform the works to a tsv (tabular data) while anonymizing their IDs.
You have to specidy the main folder containing works folder "works" and "expert_works.json".
This will create 3 files: "expert_works_anonym.json" with the new works IDs, "works_id_mapping.json" (a mapping from new IDs to OpenAlex IDs), and works.tsv (id \t title \t abstract).
You can specify the languages to translate from if detected in the title and/or the abstract.
If not specified, no translation.
```sh
python exec/build_tsv_works.py openalex --t fr,ar
```

**Interest processing**

To extract the interests into a tsv file.
It will create "profile/interests.tsv" (id \t keywords) and experts_interests.json (each expert and its keywords IDs).
```sh
python exec/build_tsv_interests.py profile --int examples/info/experts_interests.txt 
```

### 2. Representation modeling

In here, we discuss how to create models for both subjects and experts.

#### 2.1. Field taxonomy preparation

In case we need a new taxonomy, we just need to prepare a mapping file and encode the train subjects like we did in data collection and preparation.

#### 2.2. Subject modeling

Using a training dataset and a PLM for text encoding, we can train a multilabel encoder.
Since we want to train just a multilabel classification head, we first start by generating embeddings and store them on disk. 
This helps optimize memory (no model and tokenizer) and time (tokenization takes time and repeating it each epock/step is time consumable). 
First you have to prepare a config file similar to [generate_embedding.json](./examples/subject/generate_embedding.json).

```sh
python exec/generate_embeddings.py examples/subject/generate_embedding.json 

```

You can check Kaggle notebook [arxiv-cs-generate-embeddings-2gpus](https://www.kaggle.com/code/kariminf/arxiv-cs-generate-embeddings-2gpus) for a tutorial how embeddings were generated from [cs-papers-arxiv-multilabel](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-multilabel) and stored to [cs-papers-arxiv-embeddings](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-embeddings). 

#### 2.3. Expert modeling


### 3.Matching

### 3.2. Subject representation

### 3.3. Expert representation

### 3.1. Expert scoring
