from pathlib import Path

import os
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments
from typing import Union

from base_vlm import BaseVLM
from data import VQADataset

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class CustomDataCollator:
    def __init__(self, processor, use_images = True):
        self.processor = processor
        self.use_images = use_images

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Get max sequence length
        max_length = max(f["input_ids"].shape[0] for f in features)

        def pad_tensor(tensor, pad_value):
            return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

        input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=self.processor.tokenizer.eos_token_id) for f in features])
        attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
        labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
        if self.use_images:
            pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

class VQADatasetForTraining(Dataset):
    def __init__(self, dataset: VQADataset, processor: AutoProcessor, use_images = True):
        self.dataset = dataset
        self.processor = processor
        self.features = ["image", "question", "answer"]
        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.use_images = use_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        if self.use_images:
            image = Image.open(item["image_path"]).convert("RGB")
    
        # Prepare input text in chat format
        if self.use_images:
            input_message = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": item["system"]}
                        ]
                    },
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
        else:
            input_message = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": item["system"]}
                        ]
                    },
                    {"role": "user", "content": [{"type": "text", "text": item["question"]}]}]

        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        full_text = f"{prompt} {item['answer']}{self.processor.tokenizer.eos_token}"

        if self.use_images:
            inputs = self.processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
                max_length=1000
            )

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)
        else:
            inputs = self.processor.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
                max_length=1000
            )

            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

        # Get answer length
        answer_ids = self.processor.tokenizer(item["answer"], padding_side="left", return_tensors="pt", truncation=True, max_length=1000).input_ids.squeeze(0)
        answer_len = len(answer_ids)

        # Prepare labels: mask everything except the answer tokens
        labels = input_ids.clone()
        labels[:-answer_len] = -100  # only keep loss on answer

        # Ensure EOS token is at the end of the sequence
        if input_ids[-1] != self.processor.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([self.processor.tokenizer.eos_token_id])])
            attention_mask = torch.cat([attention_mask, torch.tensor([1])])
            labels = torch.cat([labels, torch.tensor([self.processor.tokenizer.eos_token_id])])

        if self.use_images:
            return {
                "input_ids": input_ids.long(),
                "attention_mask": attention_mask.long(),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": labels.long(),
            }
        else:
            return {
                "input_ids": input_ids.long(),
                "attention_mask": attention_mask.long(),
                "labels": labels.long(),
            }

def train(
    data_dir: Union[Path, None] = None,
    train_dataset_name: str = "train_vlm_diagram_QA",
    output_dir: str = "vlm_sft_QA_2.2B",
    num_train_epochs: int = 0.2,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    #lora_dropout: float = 0.1,
    num_workers: int = 1,
    use_images = True
):
    """
    Fine-tune a VLM model using LoRA.

    Args:
        model_name: Name of the base model to fine-tune
        data_dir: Directory containing the dataset
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
    """
    vlm = BaseVLM()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    processor = vlm.processor
    model = vlm.model

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.train()

    # Prepare datasets
    train_dataset = VQADataset(train_dataset_name, data_dir)
    train_dataset = VQADatasetForTraining(train_dataset, processor, use_images = use_images)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    collator = CustomDataCollator(processor, use_images = use_images)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

    # Close TensorBoard writer
    writer.close()

    return model, processor

def test_model(ckpt_path: str):
    import random

    testset = VQADataset("valid_vlm_image")
    vlm = BaseVLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    vlm.model = PeftModel.from_pretrained(vlm.model, ckpt_path, local_files_only=True).to(vlm.device)

    d = random.sample(testset.qa_pairs, 10)

    image_path = [str(o["image"]) for o in d]
    prompts = [o["system"] for o in d]
    questions = [o["user"] for o in d]
    answers = [o["response"] for o in d]

    responses = vlm.answer(image_path, prompts, questions, temperature = 0, use_images=True)

    for q, r, a in zip(questions, responses, answers):
        print(f"Q: {q}")
        print(f"\nR: {r}")
        print(f"A: {a}\n\n")

if __name__ == "__main__":
    train()
    test_model("./vlm_sft_QA_2.2B")