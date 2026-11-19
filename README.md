# BPE Tokenizers and Sentence Transformers

A Jupyter notebook exploring two related ideas in natural language processing:

1. How byte pair encoding (BPE) tokenizers split text into token IDs and pieces.
2. How a pretrained Sentence Transformer converts sentences into dense vectors.

The examples are collected in [`GDG_Event.ipynb`](GDG_Event.ipynb), which can be opened in Google Colab using the badge in the notebook.

## Notebook contents

### Compare pretrained tokenizers

The notebook uses [`tiktoken`](https://github.com/openai/tiktoken) to encode sample strings with four named encodings:

- `gpt2`
- `p50k_base`
- `cl100k_base`
- `o200k_base`

For each string, it displays the token IDs, decoded token pieces, and token count in a pandas DataFrame. The sample inputs include ordinary English words, a long word, repeated characters, Python code, an emoji, and text containing Malayalam. This is a side-by-side tokenization demonstration; it does **not** measure runtime, memory, or tokenizer performance.

### Build a small BPE merge example

The notebook starts with a tiny corpus (`low`, `lower`, `lowest`, `new`, `newer`, `newest`, `wide`, `wider`, and `widest`). It represents each word as characters followed by an `</w>` end-of-word marker, counts adjacent symbol pairs, and merges the most frequent pair five times.

This is an educational illustration of pair counting and merging, not a complete tokenizer. It does not train a production vocabulary, implement an encoder/decoder for arbitrary text, or reproduce any of the pretrained `tiktoken` encodings.

### Create sentence embeddings

The final section loads `sentence-transformers/all-MiniLM-L6-v2` through Sentence Transformers and encodes four short example sentences with normalized embeddings. It prints the resulting array shape and also encodes a single greeting.

The notebook demonstrates embedding generation. It does not calculate pairwise similarity scores, build a search index, train a model, or evaluate embedding quality.

## Run the notebook

### Google Colab

Open `GDG_Event.ipynb` in Colab using the badge at the top of the notebook. In a fresh Colab runtime, install the dependencies in a code cell before running the notebook:

```python
%pip install tiktoken pandas sentence-transformers
```

Then use **Runtime → Run all**. The first Sentence Transformer run downloads the pretrained model, so it needs an internet connection and may take longer than later runs.

### Local Jupyter

Use Python with Jupyter installed, then install the same packages and launch the notebook:

```bash
python -m pip install jupyterlab tiktoken pandas sentence-transformers
jupyter lab GDG_Event.ipynb
```

No API keys or project-specific configuration are used by the notebook. The model files are downloaded by Sentence Transformers on first use and cached by the local Hugging Face setup.

## Repository contents

| File | Purpose |
| --- | --- |
| [`GDG_Event.ipynb`](GDG_Event.ipynb) | Tokenizer comparisons, a toy BPE merge walkthrough, and sentence embedding examples |
| `README.md` | Project overview and instructions for running the notebook |

## Notes

- Token IDs and token counts depend on the selected encoding; they are not directly comparable as a speed or quality benchmark.
- The hand-written BPE section operates only on its small example corpus and five merge steps.
- Sentence embeddings are vectors, not generated text. The notebook does not yet demonstrate semantic search or similarity ranking.
- No separate requirements file is included; install the packages listed above in the notebook environment.
