# Latent Language

Train independent language autoencoders, give them names, and combine the encoder from one with the decoder from another. Merging copies the selected components into a self-contained model file. It does not fit an alignment map or update any weights.

## Install

Use a Python 3.10+ environment with a hardware-compatible PyTorch installation (>=2.6,<3), then run:

```bash
python -m pip install -r requirements.txt
```

## Train and name models

```bash
python main.py train --language english --name english_v1
python main.py train --language spanish --name spanish_v1
```

English and Spanish are currently supported (`en` and `es` also work). Each model starts from independently randomized weights and records its seed. Names can contain letters, numbers, hyphens and underscores. Existing names are protected from accidental overwrites.

By default, the first training command generates a small **synthetic toy corpus** automatically. It is a controlled experiment, not broad English or Spanish training. To use real text, pass `--data /path/to/data`. That folder needs `train.en.txt` or `train.es.txt` with one sentence per line, plus `val.jsonl` containing rows with an `en` or `es` field for the chosen language. Keep validation content out of training. Long inputs are rejected instead of silently truncated.

## Choose an encoder and decoder

```bash
python main.py merge --encoder english_v1 --decoder spanish_v1 --name english_to_spanish
```

Select models by their saved names, so multiple versions of each language can coexist. The input uses the encoder's tokenizer and the output uses the decoder's tokenizer. The merged file includes both selected components and both tokenizers; it does not depend on the original training folders remaining in place.

## Generate text

```bash
python main.py generate --model english_to_spanish --text "The red dog follows the blue cat."
```

Use an original model name to check same-language reconstruction:

```bash
python main.py generate --model english_v1 --text "The red dog follows the blue cat."
```

List the saved models and their languages:

```bash
python main.py list
```

## Storage and training options

Models are saved under `models/NAME/`. Each training epoch saves `last.pt` for resuming and updates `best.pt` if masked validation loss improves. Merging uses `best.pt` from each selected model and saves `merged.pt` under the new name.

To choose a shared cluster storage location, put `--models-dir` before the action, using the same directory for training, merging and generation:

```bash
python main.py --models-dir /shared/my-models train --language english --name english_v1
```

Training defaults to 15 epochs and a batch size of 128. Override these with `--epochs` and `--batch-size`. Use `--seed` for a chosen random seed. To resume a named run to 30 total epochs:

```bash
python main.py train --language english --name english_v1 --resume --epochs 30
```

Resuming restores saved training settings; the original data must remain available. Recovery starts at the last completed epoch. Do not run simultaneous trainers under the same name or concurrently generate the default toy data for the first time. Generate it through the first training run before starting the other language.

The code uses a GPU when visible, otherwise CPU; `--device cuda` makes GPU availability a requirement for training or generation. Scheduling and environment activation are left to the cluster user. No notebook or scheduler scripts are included.

## Python interface

The same operations can be called from Python:

```python
from latentlab.app import train_named, merge_named, generate_named

train_named(language="english", name="english_v1")
train_named(language="spanish", name="spanish_v1")
merge_named(encoder="english_v1", decoder="spanish_v1", name="english_to_spanish")
text = generate_named("english_to_spanish", "The red dog follows the blue cat.")
```

For repeated generation, load once with `MergedModel("models/english_to_spanish/merged.pt")` and call its `generate(text)` method.

## Experiment scope

The default architecture uses four encoder and four decoder layers, width 256, four attention heads and eight latent slots. Each model reconstructs text from 30% token masking through its bottleneck. Inputs are limited to 64 tokens including start/end tokens. The synthetic corpus has 20,000 training events per language and separate held-out data.

Attaching compatible tensor shapes does not guarantee meaningful translation. Independently learned latent coordinates can differ. Verify reconstruction first, then inspect swaps. A second English model with a different seed provides a same-language control. There is no explicit alignment stage.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover training/resuming, direct evaluation, and saving/reloading merged components. See `TESTING.md` for validation scope.
