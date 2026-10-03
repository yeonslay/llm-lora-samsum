import gc
import hashlib
import json

import evaluate
import pandas as pd
import torch
from bert_score import score as bertscore
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm

from experiment import format_prompt, get_experiment


def generate_predictions(model, tokenizer, test_dataset):
    predictions = []
    for start in tqdm(range(0, len(test_dataset), 4), desc="Generating summaries"):
        batch = test_dataset[start:start + 4]
        inputs = tokenizer(
            [format_prompt(dialogue) for dialogue in batch["dialogue"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # generate() returns the padded input followed by newly generated tokens.
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        predictions.extend(
            text.strip() for text in tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )
    return predictions


def main():
    experiment = "Experiment10"  # Use "Base" to evaluate the base model.
    config = get_experiment(experiment)
    set_seed(config["seed"])
    adapter_dir = config["run_dir"] / "final_adapter"
    evaluation_dir = config["run_dir"] / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError(f"Existing evaluation: {evaluation_dir}. Move it before re-evaluating.")
    tokenizer_source = config["base_model"] if experiment == "Base" else str(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"  # Preserve the original experiment's truncation policy.
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], dtype=torch.float16, device_map={"": 0},
    )
    if experiment != "Base":
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    test_dataset = load_dataset(config["dataset"])["test"]
    predictions = generate_predictions(model, tokenizer, test_dataset)
    references = test_dataset["summary"]
    assert len(predictions) == len(references)

    # Free the 8B model before loading the BERTScore encoder on the same GPU.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rouge = evaluate.load("rouge").compute(
        predictions=predictions, references=references, use_stemmer=True,
    )
    (precision, recall, f1), score_hash = bertscore(
        cands=predictions,
        refs=references,
        model_type="roberta-large",
        lang="en",
        idf=True,
        batch_size=16,
        device="cuda:0",
        return_hash=True,
    )
    evaluation_dir.mkdir(parents=True)
    prediction_path = evaluation_dir / "predictions.csv"
    pd.DataFrame({
        "id": test_dataset["id"],
        "prediction": predictions,
        "reference": references,
    }).to_csv(prediction_path, index=False)
    scores = {
        "experiment": experiment,
        **rouge,
        "bertscore_model": "roberta-large",
        "bertscore_precision": precision.mean().item(),
        "bertscore_recall": recall.mean().item(),
        "bertscore_f1": f1.mean().item(),
    }
    pd.DataFrame([scores]).to_csv(evaluation_dir / "scores.csv", index=False)
    metadata = {
        **config,
        "run_dir": str(config["run_dir"]),
        "test_samples": len(test_dataset),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "max_input_length": 512,
        "max_new_tokens": 128,
        "batch_size": 4,
        "do_sample": False,
        "num_beams": 1,
        "use_stemmer": True,
        "bertscore_idf": True,
        "bertscore_hash": score_hash,
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
    }
    (evaluation_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
