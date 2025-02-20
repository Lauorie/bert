import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import torch
import datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torchkeras import KerasModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 100
    patience: int = 5
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42

class TextClassifier:
    def __init__(
        self,
        model_path: str,
        config: TrainingConfig,
        id2label: Optional[Dict[int, str]] = None
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(id2label) if id2label else 16
        )
        if id2label:
            self.model.config.id2label = id2label
        
    @staticmethod
    def load_data(json_path: str) -> List[Dict]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_dataset(self, data: List[Dict]) -> datasets.Dataset:
        data_list = [
            {
                "text": "\n".join([f"{conv['role']}: {conv['content']}" 
                                 for conv in item["conversations"]]),
                "label": item["task_type"]
            }
            for item in data
        ]
        df = pd.DataFrame(data_list)
        ds = datasets.Dataset.from_pandas(df)
        ds = ds.shuffle(seed=self.config.seed)
        return ds.rename_columns({"text": "text", "label": "labels"})

    def tokenize_dataset(self, dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset.map(
            lambda example: self.tokenizer(
                example["text"],
                max_length=self.config.max_length,
                truncation=True,
                padding='max_length'
            ),
            batched=True,
            batch_size=100,
            num_proc=self.config.num_workers
        )

    def create_dataloaders(self, dataset: datasets.Dataset) -> Dict[str, DataLoader]:
        dataset.set_format(
            type="torch",
            columns=["input_ids", 'attention_mask', 'token_type_ids', 'labels']
        )
        
        # 分割数据集
        train_val, test = dataset.train_test_split(
            test_size=self.config.test_size
        ).values()
        train, val = train_val.train_test_split(
            test_size=self.config.val_size
        ).values()

        def collate_fn(examples):
            return self.tokenizer.pad(examples)

        dataloader_kwargs = {
            "batch_size": self.config.batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.config.num_workers,
            "pin_memory": True
        }

        return {
            "train": DataLoader(train, shuffle=True, **dataloader_kwargs),
            "val": DataLoader(val, **dataloader_kwargs),
            "test": DataLoader(test, **dataloader_kwargs)
        }

    def setup_training(self, train_dataloader: DataLoader):
        # 优化器设置
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        num_training_steps = len(train_dataloader) * self.config.epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def train(self, dataloaders: Dict[str, DataLoader]):
        optimizer, scheduler = self.setup_training(dataloaders["train"])

        class CustomStepRunner(KerasModel.StepRunner):
            def __call__(self, batch):
                out = self.net(**batch)
                loss = out.loss
                preds = out.logits.argmax(axis=1)
                
                if self.optimizer and self.stage == "train":
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                all_loss = self.accelerator.gather(loss).sum()
                labels = batch['labels']
                acc = (preds == labels).sum() / ((labels > -1).sum())
                all_acc = self.accelerator.gather(acc).mean()
                
                step_losses = {
                    f"{self.stage}_loss": all_loss.item(),
                    f'{self.stage}_acc': all_acc.item()
                }
                
                step_metrics = {}
                if self.stage == "train":
                    step_metrics['lr'] = (self.optimizer.state_dict()['param_groups'][0]['lr'] 
                                        if self.optimizer else 0.0)
                
                return step_losses, step_metrics

        KerasModel.StepRunner = CustomStepRunner

        keras_model = KerasModel(
            self.model,
            loss_fn=None,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

        try:
            keras_model.fit(
                train_data=dataloaders["train"],
                val_data=dataloaders["val"],
                ckpt_path='bert_cls.pt',
                epochs=self.config.epochs,
                patience=self.config.patience,
                monitor="val_acc",
                mode="max",
                plot=True,
                wandb=False,
                quiet=False
            )
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def save_model(self, output_path: str):
        try:
            self.model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            logger.info(f"Model saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

def main():
    # 配置
    model_path = "/root/app/models/bert-base-multilingual-cased"
    json_path = "/root/app/rag_data/qwen_bench/data_clsed/qwen_bench_50k_6809_cls.json"
    output_path = "qwen_bench_bert"
    
    id2label = {
        0: "知识问答", 1: "角色扮演", 2: "写作", 3: "开放问答",
        4: "数学问答", 5: "代码编程", 6: "总结", 7: "信息提取",
        8: "文本改写", 9: "翻译", 10: "逻辑推理", 11: "文本分类",
        12: "阅读理解", 13: "情感分析", 14: "文本纠错", 15: "其他"
    }

    config = TrainingConfig()
    classifier = TextClassifier(model_path, config, id2label)

    # 数据处理和训练
    data = classifier.load_data(json_path)
    dataset = classifier.prepare_dataset(data)
    encoded_dataset = classifier.tokenize_dataset(dataset)
    dataloaders = classifier.create_dataloaders(encoded_dataset)
    
    # 训练模型
    classifier.train(dataloaders)
    
    # 保存模型
    classifier.save_model(output_path)

if __name__ == '__main__':
    main()
