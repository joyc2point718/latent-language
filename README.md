# Latent Language Lab

A small, from-scratch experiment: train English and Spanish denoising autoencoders independently, then connect one model's encoder to another model's decoder. Compare direct swapping with a fitted latent coordinate map.

**Start here:** upload this repository to GitHub and open `notebooks/colab.ipynb` in Colab. No pretrained multilingual weights or external datasets are used by the default experiment.

## 1. Upload to GitHub

1. Unzip the downloaded archive on your computer.
2. Create a repository named `latent-language-lab` on GitHub. A public repository is easiest to clone from Colab without configuring authentication. Review the files before publishing.
3. On the repository page choose **Add file → Upload files** (or the new repository's “uploading an existing file” link).
4. Upload the **contents** of the extracted `latent-language-lab` folder. `README.md`, `requirements.txt`, and `pyproject.toml` should appear at the repository root, alongside `latentlab`, `tests`, and `notebooks`. Do not upload the ZIP itself or nest everything inside another folder.
5. Commit the uploaded files. The optional `.github` and `.gitignore` files may be hidden by your file browser; include them if possible. The code runs without the GitHub Actions workflow.

You do not need Git installed for this browser-based route. Do not upload your trained checkpoints or data to GitHub; save those in Drive.

## 2. Open in Colab

Visit https://colab.research.google.com/ → **File → Open notebook → GitHub**, paste your repository URL, and select `notebooks/colab.ipynb`.

Select a GPU under **Runtime → Change runtime type**; choose A100 when available. Run the notebook cells in order. Paste your own repository URL into the first code cell. The notebook mounts Google Drive so checkpoints survive runtime deletion.

A Colab notebook is a frontend to a temporary machine. `git clone` downloads code onto that machine. `%pip install -r requirements.txt` installs extra dependencies. `%pip install -e . --no-deps` makes the local package importable and registers its CLI. Repeat setup after a fresh runtime. Restart the runtime if Colab asks after dependency installation, then rerun setup.

Minimal setup in your own notebook:

```python
!git clone https://github.com/YOUR_USERNAME/latent-language-lab.git
%cd /content/latent-language-lab
%pip install -r requirements.txt
%pip install -e . --no-deps

import torch
from latentlab.model import Autoencoder, Config
print(torch.__version__, torch.cuda.is_available())
```

Keep Colab's CUDA-enabled PyTorch. This project requires PyTorch 2.6 or newer (below 3.0); the notebook checks this before training. **Do not install the CPU wheel in Colab.** `requirements.txt` pins the direct additional dependencies. It is not a complete transitive lockfile; the notebook saves `pip freeze` and GPU information with your experiment.

For a local CPU environment instead:

```bash
python -m venv .venv
# Activate the environment using your operating system's usual command.
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

## 3. Run the experiment

These commands run from the repository root. The supplied notebook redirects `--out` into Google Drive instead of the temporary `runs` directory.

```bash
python -m latentlab.cli prepare --out data/toy
python -m latentlab.cli train --data data/toy --lang en --seed 1 --out runs/en_a
python -m latentlab.cli train --data data/toy --lang en --seed 2 --tokenizer runs/en_a/tokenizer.json --out runs/en_b
python -m latentlab.cli train --data data/toy --lang es --seed 3 --out runs/es_a
```

Default architecture: 4 encoder and 4 decoder layers, width 256, 4 attention heads, 8 latent slots, maximum 64 tokens. Eight learned queries attend to the encoder output; the decoder sees only the resulting slots. Embeddings are tied within each monolingual model, never across models. The tokenizer is BPE trained only on that language's training text. The controlled corpus has a small vocabulary, so the actual vocabulary is smaller than the requested 8,000 cap.

The starter uses **30% independent token masking**, rather than span masking. Teacher-forced cross-entropy trains the decoder; evaluation uses greedy free-running generation. Training uses AdamW at 3e-4, gradient clipping, batch size 128 and 15 epochs by default, with BF16 on supported GPUs. This initial implementation uses a constant learning rate; it does not yet implement warmup, gradient accumulation, diffusion or a VAE. GPU time depends on runtime and chosen settings; no timing estimate is assumed.

A 256-wide model is intentionally modest for an A100. Establish good reconstruction first. To try a larger run use `--width 512 --heads 8 --layers 6`; keep matching latent dimensions across models being swapped. Use a new output directory for changed configurations.

### Reconstructions and controls

```bash
python -m latentlab.cli evaluate --source runs/en_a/best.pt --target runs/en_a/best.pt --pairs data/toy/val.jsonl --out runs/reconstruction.json
python -m latentlab.cli evaluate --source runs/en_a/best.pt --target runs/en_a/best.pt --pairs data/toy/val.jsonl --shuffle-latents --out runs/shuffled.json
python -m latentlab.cli evaluate --source runs/en_a/best.pt --target runs/en_b/best.pt --pairs data/toy/test.jsonl --out runs/english_swap.json
python -m latentlab.cli evaluate --source runs/en_a/best.pt --target runs/es_a/best.pt --pairs data/toy/test.jsonl --out runs/spanish_swap.json
```

Check Spanish self-reconstruction too by using `es_a/best.pt` for both source and target. Use validation, not final test scores, to choose architecture and training duration. Use `--limit 100` for quick validation previews. Evaluation saves chrF, exact string match, metadata, and a neighboring `.predictions.jsonl` file with inputs, references and generated outputs. These are not a full semantic evaluation suite.

### Frozen-model alignment

```bash
python -m latentlab.cli align --source runs/en_a/best.pt --target runs/es_a/best.pt --pairs data/toy/align.jsonl --kind orthogonal --n 1000 --out runs/en_es_orthogonal.pt
python -m latentlab.cli evaluate --source runs/en_a/best.pt --target runs/es_a/best.pt --bridge runs/en_es_orthogonal.pt --pairs data/toy/test.jsonl --out runs/spanish_aligned.json
```

Repeat with `--kind linear` for ridge-regularized linear alignment. Use `--n 100` for a smaller alignment set. Add `--shuffle-pairs` when fitting as a negative control. Fit the same bridges between `en_a` and `en_b` as a same-language control. Repeat evaluations on `ood.jsonl` for held-out agent/action combinations.

The bridge is one shared `width × width` feature map plus mean centering. **It assumes corresponding slot indices.** Independently learned slots may not correspond, so a failed bridge is not evidence of absent shared meaning. Slot assignment, slot-mixing maps and nonlinear bridges are later extensions. All monolingual weights stay frozen during fitting. Alignment uses bilingual pairs and must be reported as supervised alignment, even though initial autoencoder training was monolingual.

## Checkpoints and interruptions

**Storage recommendation:** GitHub for source code, Google Drive for active checkpoints, and optionally Hugging Face for sharing finished models. The default controlled-corpus models have about 7.75 million parameters each: approximately 31 MB of float32 weights, or 93 MB per checkpoint including Adam optimizer state. Keeping `best.pt` and `last.pt` for three models is roughly 560 MB plus data/results. Larger vocabularies and architectures increase this. BF16 computation does not shrink these checkpoints because master parameters and optimizer state remain float32. Only the current best and latest checkpoints are retained; this does not create a new permanent file every epoch. Atomic writes briefly require extra space for a temporary checkpoint.


Each completed epoch atomically writes `last.pt` (optimizer and RNG state included). `best.pt` is selected by masked validation loss. Files include architecture, tokenizer, language, training/validation hashes and seed. `environment.json` records the runtime and training arguments, and `metrics.jsonl` records losses.

To continue a 15-epoch run to 30 total epochs:

```bash
python -m latentlab.cli train --data data/toy --lang en --seed 1 --out runs/en_a --epochs 30 --resume
```

Repeat the original architecture, batch-size, learning-rate and mask-rate flags when resuming nondefault runs. An interrupted epoch restarts from the last completed epoch. This is **epoch-boundary resumption**, not per-batch resumption. GPU floating-point kernels can still produce run-to-run variation; saved seeds do not guarantee bitwise identical results across devices or library versions. Do not run two trainers into one directory.

## Data and scientific scope

`prepare` creates 20,000 training events, 1,000 validation pairs, 1,000 alignment pairs, 1,000 test pairs and 500 held-out-combination pairs. All events are unique across these splits. The training files have independent language-specific order and contain no paired IDs. Both languages nevertheless express the same training events: this is a **matched-content experiment**, not independently collected natural corpora.

The grammar uses eight animals, six transitive actions, five colors, past/present tense and negation. It is deliberately restricted, with one canonical sentence form per language. The held-out split excludes dog/greet and fox/help combinations from all ordinary splits. Examples are generated locally; no dataset download is required.

For a later natural-text run, create a new data directory with:

- `train.en.txt` and `train.es.txt`: one nonempty sentence per line, independently shuffled.
- `val.jsonl`, `align.jsonl`, `test.jsonl`: one `{"en": "...", "es": "..."}` object per line.
- Optional `ood.jsonl` with the same format.

Use separately reserved OPUS-100 English–Spanish pairs as alignment/evaluation data. Deduplicate and split **before** training tokenizers; exclude evaluation content from both training corpora. The starter deliberately rejects overlength examples rather than silently truncating them. Filter those sentences or raise `--max-len`. Natural-data acquisition and filtering are not automated in this version.

A failed swap can reflect incompatible coordinates, slot organization, weak reconstruction or a decoder that ignores its input. Fluent output can still change meaning. Inspect role reversal and negation errors, compare shuffled-latent scores, and repeat the key experiments over at least three seeds before drawing conclusions. This starter implements the mechanics; it does not demonstrate semantic alignment in advance.

## Python interface and component export

```python
import torch
from latentlab.experiment import load_model
from latentlab.data import encode_texts, padded

encoder_model, en_tokenizer, _ = load_model("runs/en_a/best.pt", "cuda")
decoder_model, es_tokenizer, _ = load_model("runs/es_a/best.pt", "cuda")
ids = padded(encode_texts(en_tokenizer, ["The red dog follows the blue cat."], 64), "cuda")
with torch.no_grad():
    z = encoder_model.encode(ids)       # [batch, slots, width]
    output = decoder_model.generate(z) # direct swap; use apply_map(z, bridge) for alignment
print(es_tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
```

CLI evaluation loads both full autoencoder checkpoints but calls only the source encoder and target decoder. For separate component files:

```bash
python -m latentlab.cli export --checkpoint runs/en_a/best.pt --out runs/en_a/components
```

`encoder.pt` and `decoder.pt` contain only the respective component weights plus their configuration/tokenizer. The full-checkpoint loader is `load_model`; exported component files use `state_dict` and can be loaded into an `Autoencoder(Config(**file['config']))` using `load_state_dict(..., strict=False)`. Only call the component you loaded, because the unused half remains randomly initialized. Encoder and decoder copies include their own embeddings so future modules can be attached independently.

## References

- BART denoising objective: https://aclanthology.org/2020.acl-main.703/
- Unsupervised machine translation: https://aclanthology.org/D18-1549/
- Platonic Representation Hypothesis: https://proceedings.mlr.press/v235/huh24a.html
- OPUS-100: https://opus.nlpl.eu/OPUS-100
- PyTorch mixed precision: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html

This code is a small custom PyTorch model inspired by denoising encoder–decoder architectures; it is not an implementation of the full BART pretraining recipe.
