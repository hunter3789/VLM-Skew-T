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

#processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_OFFLINE"] = "1"

def load(model_name: str = "vlm_model") -> BaseVLM:
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vlm.model = PeftModel.from_pretrained(vlm.model, model_path).to(vlm.device)
    vlm.model.eval()

    return vlm


def custom_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }

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
            '''
            input_message = [
                    {
                        "role": "system",
                        "content": [
                        #   {"type": "text", "text": "You are a weather forecaster analyzing atmospheric soundings shown in Skew-T log-P diagrams.\n\n- Lower layer: 1000–850 hPa\n- Mid layer: 850–500 hPa\n- Upper layer: 500–250 hPa\n\nDiagram legend:\n- Red line: temperature\n- Green line: dew point temperature\n- Shaded blue area: CAPE (Convective Available Potential Energy)\n- Shaded yellow area: CIN (Convective Inhibition)\n\nMeteorological interpretation tips:\n- When the red and green lines are close, the atmosphere is moist.\n- The LFC (Level of Free Convection) is the lowest point of the blue area.\n- The EL (Equilibrium Level) is the highest point of the blue area.\n- Wind barbs are displayed on the right side. If they rotate clockwise with height, it indicates veering winds; if counterclockwise, it indicates backing.\n\nPlease describe the atmospheric profile based on the provided Skew-T log-P diagram and the brief atmospheric sounding summary. Reason carefully, and conclude with a precipitation probability category: Low, Moderate, High, or Very High."}
                            {"type": "text", "text": "You are a weather forecaster analyzing atmospheric soundings shown in Skew-T log-P diagrams.\n\n- Lower layer: 1000–850 hPa\n- Mid layer: 850–500 hPa\n- Upper layer: 500–250 hPa\n\nDiagram legend:\n- Red line: temperature\n- Green line: dew point temperature\n- Shaded blue area: CAPE (Convective Available Potential Energy)\n- Shaded yellow area: CIN (Convective Inhibition)\n\nMeteorological interpretation tips:\n- When the red and green lines are close, the atmosphere is moist.\n- The LFC (Level of Free Convection) is the lowest point of the blue area.\n- The EL (Equilibrium Level) is the highest point of the blue area.\n- Wind barbs are displayed on the right side. If they rotate clockwise with height, it indicates veering winds; if counterclockwise, it indicates backing.\n\nPlease describe the atmospheric profile based on the provided Skew-T log-P diagram. Reason carefully, and conclude with a precipitation probability category: Low, Moderate, High, or Very High."}
                        #   {"type": "text", "text": "You are a weather forecaster analyzing atmospheric soundings shown in Skew-T log-P diagrams.\n\n- Lower layer: 1000–850 hPa\n- Mid layer: 850–500 hPa\n- Upper layer: 500–250 hPa\n\nDiagram legend:\n- Red line: temperature\n- Green line: dew point temperature\n- Shaded blue area: CAPE (Convective Available Potential Energy)\n- Shaded yellow area: CIN (Convective Inhibition)\n\nMeteorological interpretation tips:\n- When the red and green lines are close, the atmosphere is moist.\n- The LFC (Level of Free Convection) is the lowest point of the blue area.\n- The EL (Equilibrium Level) is the highest point of the blue area.\n- Wind barbs are displayed on the right side. If they rotate clockwise with height, it indicates veering winds; if counterclockwise, it indicates backing.\n\nReason carefully, and please provide the predicted precipitation probability category: Low, Moderate, High, or Very High."}
                        ]
                    },
                    #{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
                    {"role": "user", "content": [{"type": "image"}]}]
            '''
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
                           {"type": "text", "text": "First, extract numeric values from the given texts.\nNext, analyze the weather based on the numeric values from the atmospheric sounding summary. Reason carefully and explain your conclusions.\nBased on your reasoning, please provide the predicted precipitation probability category (Low, Moderate, High, or Very High)."}
                        ]
                    },
                    {"role": "user", "content": [{"type": "text", "text": item["question"]}]}]

        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        #full_text = prompt + item["answer"]  # append the answer to the prompt
        full_text = f"{prompt} {item['answer']}{self.processor.tokenizer.eos_token}"
        #print(full_text)

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
    #for d in train_dataset:
    #    print(d)
    #    break

    train_dataset = VQADatasetForTraining(train_dataset, processor, use_images = use_images)
    #for d in train_dataset:
    #    print(d)
    #    print(d['input_ids'].shape, d['attention_mask'].shape, d['labels'].shape)
    #    exit()

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


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    model.train()
    return val_loss / len(val_loader)


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )

'''
def test_model(ckpt_path: str, val_dataset: str = "valid_grader"):
    testset = VQADataset(val_dataset)

    llm = load(ckpt_path)

    benchmark_result = benchmark(llm, testset, 128)
    print(benchmark_result.accuracy)
'''

def test_model(ckpt_path: str):
    import random

    testset = VQADataset("valid_vlm_diagram_QA")
    vlm = BaseVLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    vlm.model = PeftModel.from_pretrained(vlm.model, ckpt_path, local_files_only=True).to(vlm.device)

    d = random.sample(testset.qa_pairs, 10)

    #print(d["question"])
    #image_path = str(d["image_path"])
    #print(image_path)
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
    #from fire import Fire

    #Fire({"demo_train": demo_train, "train": train, "test": test_model})
    train()
    test_model("./vlm_sft_QA_2.2B")