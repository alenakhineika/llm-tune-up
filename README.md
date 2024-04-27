# Train and fine-tune LLM

Fine-tuning of the `FLAN-T5-small` model for a question-answering task on the [yahoo_answers_qa](https://huggingface.co/datasets/yahoo_answers_qa) dataset using transformers library, and running optmized inference.

## Installation of the libraries

(Optional) create virtualenv:

```bash
python -m venv fosdemvenv && source fosdemvenv/bin/activate
```

Install requirements:


```bash
pip install nltk
pip install datasets
pip install 'transformers[torch]'
pip install tokenizers
pip install evaluate
pip install rouge_score
pip install sentencepiece
pip install huggingface_hub
```

## Model training and fine-tuning

Trigger the fine-tuning:

```bash
python /src/run-tuning.py
```

# Model inference

Use an existing model to solve a real-world problem:

```bash
python /src/run-prediction.py
```

## Resources

- [Model Card for FLAN-T5 base](https://huggingface.co/google/flan-t5-base)
- [FLAN-T5 Tutorial: Guide and Fine-Tuning](https://www.datacamp.com/tutorial/flan-t5-tutorial)