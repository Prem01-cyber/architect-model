"""
trainer.py
──────────
LoRA fine-tuning loop using HuggingFace TRL's SFTTrainer.
Handles:
  - 4-bit quantisation (QLoRA) for fitting 7B on a single GPU
  - LoRA adapter injection
  - Mixed precision training (bf16)
  - Gradient checkpointing
  - Checkpoint saving + best model selection
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers import Trainer

from training.dataset import ArchitectDataset, DataCollator

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # required for causal LM training
    return tokenizer


def load_model(model_name: str, cfg: dict) -> AutoModelForCausalLM:
    """
    Loads the base model with 4-bit QLoRA quantisation.
    Uses bfloat16 for compute dtype — better numerics than float16.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # nested quantisation saves ~0.4 bits/param
    )

    # In DDP each process owns one GPU — map model to that GPU only.
    # device_map="auto" shards across all GPUs which conflicts with DDP.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        dtype=torch.bfloat16,
    )

    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1  # tensor parallelism = 1 for single GPU

    # Prepare for k-bit training (casts non-quantised layers to float32)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model


def inject_lora(model: AutoModelForCausalLM, lora_cfg: dict) -> AutoModelForCausalLM:
    config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


def build_training_args(train_cfg: dict, output_dir: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 100),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        report_to=train_cfg.get("report_to", "none"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        seed=train_cfg.get("seed", 42),
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",  # QLoRA-optimised optimiser
        ddp_find_unused_parameters=False,
    )


class ArchitectTrainer:

    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = Path(config["training"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        logger.info(f"Loading tokenizer: {self.cfg['model']['base_model']}")
        tokenizer = load_tokenizer(self.cfg["model"]["base_model"])

        logger.info("Loading model with 4-bit QLoRA quantisation")
        model = load_model(self.cfg["model"]["base_model"], self.cfg["model"])

        logger.info("Injecting LoRA adapters")
        model = inject_lora(model, self.cfg["lora"])

        logger.info("Building datasets")
        data_cfg = self.cfg["data"]
        train_dataset = ArchitectDataset(
            data_path=data_cfg["train_path"],
            tokenizer=tokenizer,
            max_seq_length=data_cfg.get("max_seq_length", 2048),
            split="train",
            val_split=data_cfg.get("val_split", 0.05),
            seed=self.cfg["training"].get("seed", 42),
        )
        val_dataset = ArchitectDataset(
            data_path=data_cfg.get("val_path", data_cfg["train_path"]),
            tokenizer=tokenizer,
            max_seq_length=data_cfg.get("max_seq_length", 2048),
            split="val",
            val_split=data_cfg.get("val_split", 0.05),
            seed=self.cfg["training"].get("seed", 42),
        )

        training_args = build_training_args(
            self.cfg["training"],
            str(self.output_dir / "checkpoints"),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollator(),
        )

        logger.info("Starting training")
        trainer.train()

        logger.info("Saving final model")
        final_path = self.output_dir / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # Merge LoRA weights into base model for cleaner inference
        self._merge_and_save(model, tokenizer, final_path)

        logger.info(f"Training complete. Model saved to {final_path}")

    def _merge_and_save(self, model, tokenizer, adapter_path: Path) -> None:
        """
        Merges LoRA weights back into the base model.
        The merged model can then be loaded without peft.
        """
        try:
            from peft import PeftModel

            merged_path = self.output_dir / "merged"
            logger.info(f"Merging LoRA weights → {merged_path}")

            # Reload base in float16 for merge (avoids 4-bit merge issues)
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg["model"]["base_model"],
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base, str(adapter_path))
            merged = merged.merge_and_unload()
            merged.save_pretrained(str(merged_path))
            tokenizer.save_pretrained(str(merged_path))
            logger.info("Merge complete")

        except Exception as e:
            logger.warning(f"Merge failed (non-fatal): {e}")
            logger.warning("Use the adapter weights directly with peft.PeftModel")
