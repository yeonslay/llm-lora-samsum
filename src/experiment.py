from pathlib import Path


def get_experiment(name):
    """LoRA settings verified against the archived adapter_config.json files."""
    experiments = {
        "Experiment1": (["q_proj", "k_proj", "v_proj", "o_proj"], 2, 16),
        "Experiment2": (["q_proj"], 8, 16),
        "Experiment3": (["k_proj"], 8, 16),
        "Experiment4": (["v_proj"], 8, 16),
        "Experiment5": (["o_proj"], 8, 16),
        "Experiment6": (["q_proj", "v_proj"], 4, 16),
        "Experiment7": (["q_proj", "k_proj"], 4, 8),
        "Experiment8": (["q_proj", "k_proj", "v_proj", "o_proj"], 2, 4),
        "Experiment9": (["q_proj", "v_proj"], 4, 8),
        "Experiment10": (["v_proj", "o_proj"], 4, 8),
        "Experiment11": (["v_proj", "o_proj"], 8, 16),
        "Experiment12": (["v_proj", "o_proj"], 16, 32),
        "Experiment13": (["v_proj", "down_proj"], 4, 8),
    }
    if name == "Base":
        modules, rank, alpha = [], None, None
    else:
        modules, rank, alpha = experiments[name]
    return {
        "experiment": name,
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "dataset": "knkarthick/samsum",
        "target_modules": modules,
        "rank": rank,
        "alpha": alpha,
        "dropout": 0.1,
        "seed": 42,
        "run_dir": Path(__file__).resolve().parents[1] / "run" / name,
    }


def format_prompt(dialogue):
    return "\n\n".join([
        "### Instruction\nSummarize the following dialogue.",
        f"### Context\n{(dialogue or '').strip()}",
        "### Answer\n",
    ])


def format_training_example(example):
    return format_prompt(example["dialogue"]) + (example["summary"] or "").strip()
