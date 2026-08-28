# SFE-Match: Subject-Field-Expert matching

[![Project](https://img.shields.io/badge/Project-SFEMatch-0BDA51.svg?style=plastic)](https://kariminf.github.io/sfematch/)
[![License](https://img.shields.io/badge/License-Apache_2-0BDA51.svg?style=plastic)](http://www.apache.org/licenses/LICENSE-2.0)

This is an expert matching project.

## Workflow

### 1. Data collection and preparation

#### 1.1. Taxonomy collection

<!-- Arxiv taxonomies already exist in this project: cs, math, stat, bio, fin, phys.
Othor antologies can be afforded if you afford mappings from the original you used.
The following sections will discuss this. -->
SFE-Match uses a predefined research taxonomy as the representation space.
The repository currently includes arXiv taxonomies for:
- cs — Computer Science
- math — Mathematics
- stat — Statistics
- bio — Quantitative Biology
- fin — Quantitative Finance
- phys — Physics

Other taxonomies can also be used. A mapping between the source taxonomy and the target taxonomy must be provided when necessary.
Examples of taxonomy mappings are available in: [examples/subject/](./examples/subject/ccsf_mapping.json).
For the experiments reported in the paper, three taxonomies are used:
- arXiv
- ACM Computing Classification System (CCS) 2012 Level 1
- CCSF, a compact taxonomy defined for this study

#### 1.2. Scientific publication collection

ArXiv dataset can be freely downloaded from [Kaggle:arXiv](https://www.kaggle.com/datasets/Cornell-University/arxiv).

Scientific publications are used to train the subject representation model.
[arXiv](https://arxiv.org/) provides a large, openly available collection of scientific publications and is therefore used as the primary source for training data.
The arXiv metadata dataset can be obtained from: [Kaggle: arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv).


**Extract papers from an arXiv domain**

To extract papers belonging to a specific arXiv domain:

```sh
python exec/extract_arxiv_domain.py --src data/arxiv/arxiv-metadata-oai-snapshot.json --out data/arxiv/cs_papers.jsonl --dom cs 
```

Replace cs with the desired domain, such as: `math`, `stat`, `bio`, `fin` or `phys`.
The command generates a JSONL file such as: `cs_papers.jsonl`.

**Convert JSONL to TSV**

Convert the extracted dataset to TSV format:

```sh
python exec/jsonl2tsv.py --src data/arxiv/cs_papers.jsonl --dom cs
```

This generates: `cs_papers.tsv`.

**Create train/test splits**

The dataset can be divided into training and test sets while maintaining a balanced distribution of labels:

```sh
python exec/split_train_test.py data/arxiv/cs_papers.tsv  --test-size 0.2
```

This generates: `cs_papers_train.tsv` and `cs_papers_test.tsv`.
The script also reports statistics on the distribution of fields.

-------
**Separate text and labels**

To separate publication text from taxonomy labels:
```sh
python separate_text_labels.py data/arxiv/cs_papers_train.tsv
python separate_text_labels.py data/arxiv/cs_papers_test.tsv
```

For example: the file `cs_papers_train.tsv` will be separated into `cs_papers_train_text.tsv` and `cs_papers_train_labels.tsv`.

**Map labels between taxonomies**

Labels can be mapped from one taxonomy to another using:

```sh
python exec/map_labels.py data/arxiv/cs_papers_train_labels.tsv --map ./examples/subject/ccsf_mapping.json
```

For another source taxonomy, create a mapping file following the structure of: [examples/subject/ccsf_mapping.json](./examples/subject/ccsf_mapping.json)
A mapping file specifies:
- the name used for the output;
- the target taxonomy;
- the mapping associated with each source label; the indices of the corresponding target labels.

**Preprocessed arXiv datasets**

- *cs-papers-arxiv-multilabel*: 
To facilitate reproduction, preprocessed datasets for arXiv Computer Science papers are provided.
This dataset contains papers annotated according to: arXiv, ACM CCS 2012 Level 1, and CCSF.
The dataset is distributed on Kaggle under CC BY 4.0: [Kaggle:cs-papers-arxiv-multilabel](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-multilabel)

- *cs-papers-arxiv-embeddings*: 
We also provide precomputed embeddings of publication titles and abstracts generated using several PLMs, including: BERT, Sentence-BERT, SciBERT, and SPECTER2.
Providing embeddings avoids repeating the computationally expensive PLM encoding step.
The dataset is distributed on Kaggle: [Kaggle:cs-papers-arxiv-embeddings](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-embeddings)

#### 1.3. Expert data collection

Expert data collection is the most manual component of the pipeline.

**Create an expert list**

Create a text file containing one expert per line:
An example is provided in: [expert_list.txt](./examples/info/expert_list.txt).

Run:

```sh
python exec/collect_infos.py examples/info/expert_list.txt  --out examples/info/experts_info_choices.json
```

The script generates candidate identifiers and information for each expert: [experts_info_choices.json](./examples/info/experts_info_choices.json)
The correct identifiers must then be selected manually and stored in a file following the structure of: [experts_info.json](./examples/info/experts_info.json).

**Collect research interests**

Research interests are collected separately because many websites do not permit automated scraping.
Prepare a file following the structure of: [experts_interests.txt](./examples/info/experts_interests.txt).

#### 1.4. Expert publication processing

**Collect publications from OpenAlex**

[OpenAlex](https://openalex.org/) is used as the primary source for expert publications because it provides an API for retrieving scholarly metadata.

Run:

```sh
python exec/collect_works.py examples/info/experts_info.json --out data/openalex
```

The script downloads expert profiles from OpenAlex and store them under `data/openalex/profiles/`.
Then, it extracts individual works and store them under `data/openalex/works/`.
If execution is interrupted, already downloaded files can be retained and skipped during subsequent execution.

**Check publication records**

Use the following script to identify works with missing abstracts and potentially redundant records:

```sh
python exec/find_abstract_issues.py data/openalex
```

This generates: a file `data/openalex/abstract_issues_report.json` containing works without abstracts and also redundant works (probably the same work).

The reported issues can then be inspected and corrected manually.

**Index expert works**

To index works by author:

```sh
python exec/index_expert_works.py data/openalex
```

This generates: `data/openalex/indexed_works.json` containing author–work associations.
Because author names are not always represented consistently across sources, the resulting author–work associations must be manually verified.

The verified associations should then be stored in: `data/profile/expert_works.json`.

**Build the works dataset**

The verified expert–work associations can be converted into a TSV dataset while replacing the original work identifiers with anonymized identifiers:

```sh
python exec/build_tsv_works.py data/openalex --t fr,ar
```

The script generates three files: `expert_works_anonym.json`, `works_id_mapping.json` and `works.tsv`.
The files contain:
- expert_works_anonym.json — anonymized work identifiers associated with experts;
- works_id_mapping.json — mapping between anonymized identifiers and OpenAlex identifiers;
- works.tsv — tabular work data, with columns: `id`, `title` and `abstract`.

The --t option specifies languages to translate when detected in the title or abstract.
For example: `--t fr,ar` requests translation of French and Arabic text.
If --t is omitted, no translation is performed.

#### 1.5 Expert interest processing

Expert research interests can be converted to TSV format using:

```sh
python exec/build_tsv_interests.py data/profile --int examples/info/experts_interests.txt 
```

This generates: `data/profile/interests.tsv` and `experts_interests.json`.
`interests.tsv` contains a table with columns: `id` and `keywords`, while `experts_interests.json` associates each expert with the corresponding interest identifiers.

### 2. Representation modeling

The second stage generates field representations for subjects and experts.

#### 2.1. Field taxonomy preparation

If a new taxonomy is required, prepare the corresponding taxonomy mapping and encode the training subjects using the procedure described in Section 1.2.

#### 2.2. Subject modeling

A subject is represented using a PLM followed by a multi-label prediction head.

To reduce computational and storage requirements during training, PLM embeddings can first be generated and stored on disk. 
This avoids repeatedly loading the PLM and tokenizer during training.

**Generate PLM embeddings**

Prepare a configuration file following: [generate_embedding.json](./examples/subject/generate_embedding.json).
Then run:

```sh
python exec/generate_embeddings.py examples/subject/generate_embedding.json 
```

- A complete example of the embedding-generation process is available in the Kaggle notebook: [Kaggle:arxiv-cs-generate-embeddings-2gpus](https://www.kaggle.com/code/kariminf/arxiv-cs-generate-embeddings-2gpus)
- The notebook generates embeddings from: [Kaggle:cs-papers-arxiv-multilabel](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-multilabel).
- and stores the resulting embeddings in: [Kaggle:cs-papers-arxiv-embeddings](https://www.kaggle.com/datasets/kariminf/cs-papers-arxiv-embeddings). 

**Train the multi-label prediction head**

Prepare a configuration file following: [train_subject_model.json](./examples/subject/train_subject_model.json).
Then, run:

```sh
python exec/train_mlp_head.py examples/subject/train_subject_model.json 
```

**Evaluate the subject model**

Prepare a test configuration following: [test_subject_model.json](./examples/subject/test_subject_model.json).
Then run:

```sh
python exec/evaluate_mlp_head.py examples/subject/test_subject_model.json 
```

**Pretrained subject models**

- The model-training process is demonstrated in: [Kaggle:cs-fields-multilabel-mlp-model-training](https://www.kaggle.com/code/kariminf/cs-fields-multilabel-mlp-model-training).
- The resulting model is available as: [Kaggle:cs_fields_multilabel_mlp_model](https://www.kaggle.com/models/kariminf/cs_fields_multilabel_mlp_model).
- A fine-tuned Sentence-BERT version is also available:  [Kaggle:cs-fields-multilabel-sbert-ft-model-training](https://www.kaggle.com/code/kariminf/cs-fields-multilabel-sbert-ft-model-training).
- with the resulting model: [cs_fields_sbert_finetuned_model](https://www.kaggle.com/models/kariminf/cs_fields_sbert_finetuned_model).

#### 2.3. Expert modeling

Expert profiles are constructed from two sources of evidence:

1. scientific publications;
1. research interests.

Works and interests are represented using the same field-prediction model used for subjects.

**Generate embeddings**

Generate embeddings for expert works:
```sh
python exec/generate_embeddings.py examples/expert/generate_embedding_works.json 
```

Generate embeddings for expert interests:

```sh
python exec/generate_embeddings.py examples/expert/generate_embedding_interests.json 
```

**Generate field probabilities**

Use the trained subject model to generate field probabilities for works and interests:

```sh
python exec/model_works_interests.py examples/expert/model_works_interests.json
```

**Construct expert profiles**

Prepare a configuration file following: [model_experts.json](./examples/expert/model_experts.json).
Then run:

```sh
python exec/model_experts.py examples/expert/model_experts.json 
```

The resulting expert profiles combine the field representations obtained from publications and research interests.

<!-- The data folder on which we worked, can be downloaded here [Download expert matching tutorial](https://github.com/kariminf/sfematch/releases/download/0.1/data.zip)  -->

### 3.Matching

The final stage ranks experts according to their similarity to a subject.

The repository includes a real-world case study involving the assignment of jury members to final-year projects (FYPs) in a computer science institute.

**Data availability**

The raw expert and FYP data from this case study cannot be publicly released because they are proprietary to the institution.
To support reproducibility while respecting these restrictions, the repository provides derived field representations and embeddings with opaque/anonymized identifiers.
The released data allow researchers to reproduce the downstream representation and matching experiments without access to the underlying expert and FYP text.
The corresponding files include:

- field probabilities for expert works;
- field probabilities for expert interests;
- field probabilities for FYP subjects;
- aggregated expert profiles;
- manually assigned taxonomy labels;
- anonymized subject and expert identifiers;
- evaluation data and rankings where appropriate.

The case-study files can be obtained from: [Kaggle real use case](https://github.com/kariminf/sfematch/releases/download/0.1/kaggle2.zip).
The archive contains:

- `proba/`: Probabilistic field representations for: expert publications; expert interests; FYP subjects.
- `profiles/`: Expert profiles generated by combining publication and interest evidence.
- `eval/`: Expert-representation evaluation against manual annotations.
- `matching/`: Matching evaluation using the arXiv taxonomy as the reference representation.
- `matching2/`: Matching evaluation using the average similarity across the three taxonomies.
- `labels/`: Manual taxonomy annotations for FYP subjects and experts, together with anonymized jury assignments.

### 3.1. Expert scoring

Prepare a configuration file following: [match_subject_experts.json](./examples/matching/match_subject_experts.json).
The matching function can use: `cosine`, `euclidean` or `mae`.
The number of retained candidates can be controlled using `top_k`.
Set:`null` to retain all candidates for which a valid similarity score is available.

Run:

```sh
python exec/ranking.py ./examples/matching/match_subject_experts.json 
```

### 3.2. Evaluation

Prepare an evaluation configuration following: [eval_match.json](./examples/matching/eval_match.json).

Then run:

```sh
python exec/evaluate_match.py ./examples/matching/eval_match.json 
```

## Research and Reproducibility

SFE-Match was developed as part of research on interpretable expert matching for academic subject–expert assignment.
The repository separates the main components of the experimental pipeline:

- source datasets, such as arXiv and OpenAlex;
- taxonomy definitions and mappings;
- expert metadata and research interests;
- processed publication data;
- PLM embeddings;
- field-probability representations;
- expert profiles;
- matching and evaluation results.

This separation allows individual stages of the pipeline to be reproduced without necessarily repeating the entire data-collection process.

**Restricted real-world data**

The raw expert and FYP data used in the institutional case study are not distributed because they are proprietary to the institution.
Instead, we provide derived representations with opaque identifiers, including field-probability vectors and embeddings. These derived artefacts support reproduction of downstream experiments while avoiding redistribution of the underlying restricted data.
Researchers should distinguish between:

- full reproduction of the data-collection process, which requires access to the original sources and institutional data; and
- reproduction of representation and matching experiments, which can be performed using the released derived data.

**Supporting files**

The following release archives contain the files required for reproducing the tutorial and experiments.

- [Examples](https://github.com/kariminf/sfematch/releases/download/0.1/examples.zip):
Contains example input files, expert information, taxonomy mappings, and configuration files used throughout the tutorial.
- [Data](https://github.com/kariminf/sfematch/releases/download/0.1/data.zip): 
Contains the data generated by the preprocessing pipeline.
- [Kaggle real use case](https://github.com/kariminf/sfematch/releases/download/0.1/kaggle2.zip):
Contains the outputs associated with the real-world matching case study described in Section 3.
- [Kaggle tuto folder](https://github.com/kariminf/sfematch/releases/download/0.1/kaggle_tuto.zip): Contains the outputs produced from the tutorial using the released case-study data.

**Kaggle resources**

Notebooks, datasets, and models associated with the project are available in the following Kaggle collection: 
[subject_field_expert](https://www.kaggle.com/work/collections/19009995).

## Citation

If you use SFE-Match, its datasets, models, or preprocessing pipeline in your research, please cite the associated paper:

```tex
```

The complete citation will be added once the paper is published.

## License

Copyright (C) 2026 Abdelkrime Aries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
