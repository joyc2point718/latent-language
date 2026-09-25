# Validation

Four CPU tests passed with Python 3.12, PyTorch 2.6.0+cpu, tokenizers 0.23.1 and sacrebleu 2.5.1.

Tests cover:

- Separation of generated training and evaluation data.
- Training, checkpoint resumption, reconstruction, direct swaps and component export.
- Named training and resumption with saved settings, plus rejection of a language mismatch.
- Merged encoder and decoder computations matching the original selected components. A saved merged model still generates the same output after the original training folders are removed.

Python source compilation and CLI help checks passed. These checks validate software behavior with tiny models; they do not demonstrate translation quality. No full GPU training run was performed.

Run tests with `python -m unittest discover -s tests -v`.
