# Turkish Sentiment Analysis with Multilingual DistilBERT

Three-class sentiment classification for Turkish text, developed for a Natural Language Processing course at Hitit University by Omar Gouda.

**Task:** classify text as Negative, Neutral (`Notr`) or Positive.

## Approach

1. Load the [Turkish sentiment dataset](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset).
2. Sample 2,000 examples per class from its training partition.
3. Split those 6,000 examples into 4,800 training and 1,200 validation examples, stratified by label.
4. Fine-tune [multilingual DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased), selecting the checkpoint with the best validation macro F1.
5. Evaluate on a stratified sample of 3,000 examples from the dataset's separate test partition.

## Reported experiment

| Metric | Reported value |
|---|---:|
| Accuracy | 89.87% |
| Macro F1 | 85.05% |
| Weighted F1 | 90.51% |
| Negative-class F1 | 65.89% |

These are results recorded from the original Colab experiment, not a new training run performed during repository cleanup. The notebook is distributed without saved execution outputs or trained weights. Results can vary when rerun.

### Training configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Training batch size | 8 |
| Maximum sequence length | 128 tokens |
| Split / trainer seed | 42 |
| Checkpoint selection | Validation macro F1 |

## Run in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/Omar-Gouda5/turkish-sentiment-analysis/blob/main/Turkish_Sentiment_Analysis.ipynb)

1. Select a GPU runtime when available.
2. Run the notebook cells in order, including dependency installation. Restart the runtime if the environment requests it.
3. Allow access to Hugging Face to download the dataset and base model.
4. Training writes the model to `final_model/` and evaluation outputs to `results/`.

The tested historical setup used `transformers==4.46.3`. Other dependencies are listed in `requirements.txt`; the environment is not fully locked. Colab supplies PyTorch; a local notebook environment also requires a compatible PyTorch installation and Jupyter.

## Repository guide

| File | Purpose |
|---|---|
| [Notebook](Turkish_Sentiment_Analysis.ipynb) | Data preparation, fine-tuning, evaluation and sample predictions |
| [Turkish report](Turkce_Duygu_Analizi_Proje_Raporu.md) | Academic write-up |
| [Experiment summary](results/experiment_summary.md) | Recorded original metrics |
| [Confusion matrix](results/confusion_matrix.svg) | Reconstructed from the original test counts |
| [Loss chart](results/training_loss.svg) | Reconstructed from approximate, rounded report values |

## Limitations

- Lower negative-class F1 highlights uneven performance across classes; accuracy alone does not describe model quality.
- Increased validation loss in the third epoch suggests possible overfitting.
- Sampling and hardware/library differences can affect reproducibility. There is no fresh end-to-end reproduction in this cleanup.
- This is a course experiment, not a deployed sentiment service or a claim of state-of-the-art performance.
- Dataset and base-model terms remain those of their respective publishers.

## More projects

See [Omar Gouda's selected engineering and AI projects](PORTFOLIO.md) for the curated portfolio and each project's validated scope.
