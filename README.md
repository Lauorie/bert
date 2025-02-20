# Text Classification with Transformers

This repository provides a pipeline for training a text classification model using the Hugging Face `transformers` library and PyTorch. It includes data preprocessing, tokenization, model training, and model saving. The model is fine-tuned on a dataset of conversations, where each conversation is labeled with a specific task type.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Saving the Model](#saving-the-model)
- [Pipeline Overview](#pipeline-overview)
- [Dataset Preparation](#dataset-preparation)
- [Model Details](#model-details)
- [Logging](#logging)
- [License](#license)

---

## Features

- Flexible configuration using `dataclasses`.
- Preprocessing and tokenization of text data.
- Support for multi-GPU training with PyTorch.
- Built-in learning rate scheduler and optimizer.
- Early stopping and checkpoint saving during training.
- Easy loading and saving of models and tokenizers.

---

## Requirements

Ensure the following packages are installed:

- Python 3.8+
- PyTorch
- Hugging Face `transformers`
- `datasets`
- `torchkeras`
- `pandas`

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Lauorie/bert.git
   cd src
   ```

2. Install the required Python packages:

   ```bash
   pip install torch transformers datasets pandas torchkeras
   ```

---

## Usage

### Configuration

The training configuration is defined in the `TrainingConfig` class. You can modify parameters such as:

- `max_length`: Maximum token length for input sequences.
- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for the optimizer.
- `epochs`: Number of epochs to train the model.
- `val_size` and `test_size`: Proportions of validation and test splits.
- `patience`: Early stopping patience.

Example configuration:

```python
config = TrainingConfig(
    max_length=512,
    batch_size=32,
    learning_rate=2e-5,
    epochs=50,
    val_size=0.2,
    test_size=0.2,
    patience=5
)
```

### Training

1. Update the `main` function with your dataset and model paths:

   ```python
   model_path = "/path/to/pretrained-model"  # Pretrained Hugging Face model
   json_path = "/path/to/dataset.json"      # Path to dataset in JSON format
   output_path = "/path/to/save/model"      # Directory to save trained model
   ```

2. Run the script:

   ```bash
   python main.py
   ```

This will:

- Load and preprocess the dataset.
- Tokenize the text data.
- Create data loaders for training, validation, and testing.
- Train the model using the specified configuration.
- Save the trained model and tokenizer.

### Saving the Model

The trained model and tokenizer are saved to the directory specified in the `output_path`. You can then load the model for inference or further fine-tuning.

---

## Pipeline Overview

1. **Data Loading**:
   - Load data from a JSON file using the `load_data` method.
   - Each data sample includes a conversation history and a task type label.

2. **Dataset Preparation**:
   - Convert conversation data into a format suitable for training.
   - Use `datasets.Dataset` and Pandas for preprocessing.
   - Shuffle and split the dataset into training, validation, and test sets.

3. **Tokenization**:
   - Tokenize the text data using the Hugging Face tokenizer.
   - Add padding and truncation to ensure consistent input sizes.

4. **Model Training**:
   - Fine-tune a pretrained transformer model using PyTorch.
   - Use AdamW optimizer and a linear learning rate scheduler with warmup.
   - Incorporate early stopping based on validation accuracy.

5. **Model Saving**:
   - Save the fine-tuned model and tokenizer for future use.

---

## Dataset Preparation

The input dataset should be in JSON format, with the following structure:

```json
[
    {
        "conversations": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you?"}
        ],
        "task_type": 0
    },
    ...
]
```

- Each sample has a `conversations` field, which is a list of messages between the user and assistant.
- `task_type` is the label for the conversation (an integer).

---

## Model Details

The script uses a pretrained transformer model from the Hugging Face `transformers` library. By default, the `bert-base-multilingual-cased` model is used, but you can replace it with any compatible model by updating the `model_path`.

The example includes 16 task types, represented by the `id2label` mapping:

```python
id2label = {
    0: "知识问答", 1: "角色扮演", 2: "写作", 3: "开放问答",
    4: "数学问答", 5: "代码编程", 6: "总结", 7: "信息提取",
    8: "文本改写", 9: "翻译", 10: "逻辑推理", 11: "文本分类",
    12: "阅读理解", 13: "情感分析", 14: "文本纠错", 15: "其他"
}
```

You can modify the labels to suit your dataset.

---

## Logging

The script uses Python's `logging` module to provide informative logs during data processing, training, and model saving. Logs are displayed in the format:

```
2025-02-20 12:00:00 - INFO - Message
```

Error logs are also supported to handle exceptions during execution.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute it as needed.
