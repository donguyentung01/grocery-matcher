# Grocery Matcher

This project matches products from Store A to interchangeable products from
Store B. It uses local retrieval to build a small candidate set and GPT for the
final product-identity decision, including functional size and package
configuration.

## Quickstart

Everything you need, in order. Commands are for macOS/Linux (bash/zsh); a
Windows PowerShell equivalent follows each one where it differs.

### 1. Prerequisites

- Python 3.12 or newer
- An OpenAI-compatible API key (OpenAI itself, or an Azure OpenAI deployment)

### 2. Clone the repo

```bash
git clone https://github.com/donguyentung01/grocery-matcher.git
cd grocery-matcher
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Add the two dataset CSVs

The datasets aren't in the repo (they're ~230 MB combined, over GitHub's file
size limit). Get your own `grocery_store_a_items_final.csv` and
`grocery_store_b_items_final.csv` and place them in the project root
(next to `run_matcher.py`).

Each CSV must have at least these columns:

| Column | Required | Notes |
|---|---|---|
| `item_id` | yes | unique per row |
| `name` | yes | product name |
| `brand_raw` | no | improves matching if present |
| `category`, `subcategory`, `department` | no | used as context |
| `item_info` | no | free-text details |
| `sizing_comp` | no | size/pack info |

Any other columns in the file are ignored.

### 5. Add your API credentials

Copy the template and fill in your own values:

```bash
cp openai_creds.example.yaml openai_creds.yaml
```

```yaml
openai:
  endpoint: https://api.openai.com/v1   # or an Azure OpenAI endpoint
  api_key: sk-...
  deployment_name: gpt-4o-mini          # model name (OpenAI) or deployment name (Azure)
```

`openai_creds.yaml` is gitignored, so your key is never committed.

### 6. Run it

```bash
python run_matcher.py \
  --store-a grocery_store_a_items_final.csv \
  --store-b grocery_store_b_items_final.csv
```

Windows (PowerShell):

```powershell
python run_matcher.py `
  --store-a grocery_store_a_items_final.csv `
  --store-b grocery_store_b_items_final.csv
```

The default run uses:

| Setting | Default |
|---|---:|
| Output | `output.csv` |
| Credentials | `openai_creds.yaml` |
| TF-IDF candidates | 5 |
| Embedding candidates | 5 |
| TF-IDF minimum score | 0.15 |
| Embedding minimum score | 0.50 |
| GPT concurrency | 50 |
| GPT call cap | 250,000 |

Defaults can be overridden when needed:

```bash
python run_matcher.py \
  --store-a grocery_store_a_items_final.csv \
  --store-b grocery_store_b_items_final.csv \
  --output custom_output.csv \
  --gpt-concurrency 25 \
  --max-gpt-calls 10000
```

The first run downloads the `sentence-transformers/all-MiniLM-L6-v2`
embedding model (a few hundred MB) and caches it locally; subsequent runs
reuse it.

## Outputs

The run creates two files:

- `output.csv` includes IDs and names for review.
- `output_submission.csv` contains only the required ID pairs.

A sample `output.csv` from a full run is committed in this repo so you can see
the expected shape without running the pipeline yourself.


## Architecture

### 1. Load and normalize

The loader reads only fields used by matching, removes duplicate item IDs, and
normalizes:

- Product names and brands
- Nested category metadata
- Common weight and volume units
- Counts and multipacks

Raw names are retained for output and supplied to GPT alongside normalized
names and parsed quantity data. This lets GPT evaluate package configuration
without allowing ordinary size text to dominate product retrieval.

### 2. Retrieve Store B candidates

Each Store A product uses three retrieval paths:

1. Exact normalized-name lookup
2. TF-IDF word unigram/bigram similarity
3. MiniLM sentence embeddings with a FAISS HNSW index

The top five TF-IDF and top five embedding results are merged with all exact
name matches and deduplicated.

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
``` 
### 3. Classify with GPT

Every Store A product with at least one retrieved candidate receives at most
one GPT request. The request includes the normalized Store A identity, up to 20
Store B candidates, category and brand context, and retrieval-source labels.

GPT returns:

- `match` or `no_match`
- The selected Store B item ID for a match
- A structured comparison for 13 identity attributes

The attributes are:

1. Product type/subtype
2. Form/format
3. Variant/model/product line
4. Flavor/scent
5. Key ingredient/material
6. Strength/concentration
7. Intended use/function
8. Target user/species/age
9. Compatibility/application
10. Dietary/processing claims
11. Color/shade/pattern
12. Bundle composition
13. Functional size/package configuration

Functional size/package configuration covers size, capacity, count, dimensions,
and package arrangement when those details affect how the product is used.
Modest purchase-quantity differences, such as 6 oz versus 8 oz or 18 count
versus 20 count, are allowed. Functional conflicts, such as 4-gallon versus
13-gallon bags, different diaper sizes, or individual snack packs versus one
family-size bag, are rejected.

Each attribute must be `same`, `different`, `unknown`, or `not_applicable`.
The pipeline rejects a proposed match if any attribute is `different`, if the
attribute response is malformed, or if GPT selects an item that was not offered.

GPT does not return confidence scores or explanation text.

### 4. Write matches

Accepted matches are sorted by Store A item ID and written to both output
formats. Multiple Store A products may map to the same Store B product.

## Caching and resilience

- Store B embeddings and the FAISS index are cached under `cache/`.
- GPT responses are cached by deployment and the complete prompt.
- OpenAI client retries are enabled for transient request failures.

## Tests

```bash
python -m pytest -q
```

Windows (PowerShell):

```powershell
python -m pytest -q
```
