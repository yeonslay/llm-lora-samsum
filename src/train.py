import json
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from experiment import format_training_example, get_experiment


def main():
    experiment = "Experiment10"  # Change to Experiment1 ... Experiment13.
    config = get_experiment(experiment)
    run_dir = config["run_dir"]
    if run_dir.exists():
        raise FileExistsError(f"Existing run: {run_dir}. Use a fresh checkout or move it before training.")

    set_seed(config["seed"])
    dataset = load_dataset(config["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], dtype=torch.float16, device_map={"": 0},
    )
    peft_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        optim="adamw_torch",
        weight_decay=0.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        max_length=512,
        packing=False,
        completion_only_loss=False,
        seed=config["seed"],
        data_seed=config["seed"],
        report_to=["tensorboard"],
        logging_dir=str(run_dir / "runs"),
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=format_training_example,
    )
    trainer.model.print_trainable_parameters()
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start
    final_dir = run_dir / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    metadata = {
        **config,
        "run_dir": str(run_dir),
        "training_arguments": training_args.to_dict(),
        "train_runtime_seconds": elapsed,
        "train_samples": len(dataset["train"]),
        "validation_samples": len(dataset["validation"]),
        "test_samples": len(dataset["test"]),
    }
    (run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved adapter: {final_dir}; training time: {elapsed / 3600:.2f} hours")


if __name__ == "__main__":
    main()
