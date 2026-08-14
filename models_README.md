# Model Checkpoints

The trained transformer checkpoints used in this dissertation are stored in Google Drive because they exceed GitHub's normal file-size limits.

**Supporting files:**  
https://drive.google.com/drive/folders/1HGqJvGvTxvqKUopWmZzX0PrsqXXjD5DW?usp=drive_link

## Available checkpoints

| Checkpoint | Role in the project |
|---|---|
| `distilbert_response_only.pt` | DistilBERT classifier trained using the LLM response only |
| `distilbert_context_aware.pt` | DistilBERT classifier trained using the user prompt and LLM response |
| `bert_response_only.pt` | BERT classifier trained using the LLM response only; also used for the layer-probing analysis |
| `bert_context_aware.pt` | Best-performing PKU-SafeRLHF classifier; used for the historical-to-frontier generalisation evaluation |

## Notes

- The checkpoints correspond to the experiments reported in the dissertation.
- The context-aware BERT checkpoint is the model used for Experiment 10 on the 1,014-response frontier evaluation set.
- Model architectures, preprocessing, training configuration and evaluation code are documented in `notebooks/Diss1_PKU_Experiments.ipynb` and `notebooks/Diss3_Frontier_Evaluation.ipynb`.
- Large model files are intentionally kept outside the GitHub repository to keep the repository lightweight.
