# Validation performed

Validated with Python 3.12, PyTorch 2.6.0+cpu, tokenizers 0.22.2 and sacrebleu 2.5.1.

- Editable package installation succeeded.
- Three unittest tests passed, covering held-out data separation; recovery of a known orthogonal/linear coordinate transformation; and the end-to-end train → save → resume → swap → align → evaluate → component-export workflow.
- Python source and notebook code cells parsed successfully.
- The default 20,000-example data preparation and tokenizer paths ran for both languages. Longest training examples were 12 English tokens and 11 Spanish tokens including BOS/EOS, within the 64-token default.
- Default parameter counts: English 7,746,048; Spanish 7,747,840.

The tests used tiny CPU models and do not establish convergence or cross-language semantic transfer. A full training run and CUDA/BF16 execution were not performed here. Google Drive mounting and Colab's interface must be exercised in the user's own session.

Run tests from the repository root with:

```bash
python -m unittest discover -s tests -v
```
