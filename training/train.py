"""
QLoRA Fine-tuning Script for ArduPilot Parameter Tuning Model
Fine-tunes a base LLM to become an ArduPilot parameter tuning expert.

Usage:
    python train.py --model qwen2.5:14b --data ../data/dataset/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Check for required packages
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        set_seed
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType
    )
    from datasets import Dataset
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please run: pip install torch transformers peft datasets accelerate")
    sys.exit(1)


def load_dataset(data_dir: str) -> List[Dict]:
    """
    Load training dataset from JSON files.
    
    Args:
        data_dir: Directory containing training data
    
    Returns:
        List of training examples
    """
    data_path = Path(data_dir)
    examples = []
    
    # Load all JSON files in the directory
    for json_file in data_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                examples.extend(data)
            elif isinstance(data, dict):
                examples.append(data)
    
    # If no JSON files, check for .jsonl
    if not examples:
        for jsonl_file in data_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
    
    if not examples:
        print(f"Warning: No training data found in {data_dir}")
        print("Creating sample dataset...")
        examples = create_sample_dataset()
    
    return examples


def create_sample_dataset() -> List[Dict]:
    """
    Create a sample dataset for demonstration.
    """
    return [
        {
            "instruction": "Analyze this flight log and recommend PID tuning adjustments.",
            "input": """Flight Log Analysis:
- Max Roll Error: 15.2°
- Overshoot: 12%
- Settling Time: 2.3s
- Current ATC_RAT_RLL_P: 0.12
- Current ATC_RAT_RLL_I: 0.05
- Current ATC_RAT_RLL_D: 0.002
- Vibrations: 2.1g

Flight Conditions: FPV racing, indoor, light wind""",
            "output": """Analysis:
- Roll overshoot (12%) indicates P gain too high
- Settling time (2.3s) acceptable but could improve
- Vibration levels (2.1g) within acceptable limits

Recommendations:
1. Reduce ATC_RAT_RLL_P: 0.12 → 0.09 (25% reduction)
   Rationale: High P gain causes overshoot
2. Increase ATC_RAT_RLL_D: 0.002 → 0.004 (100% increase)
   Rationale: Add damping to reduce oscillation
3. Keep ATC_RAT_RLL_I: 0.05 (no change)
   Rationale: I term is in good range"""
        },
        {
            "instruction": "Analyze this flight log and recommend PID tuning adjustments.",
            "input": """Flight Log Analysis:
- Max Roll Error: 3.1°
- Overshoot: 2%
- Settling Time: 0.8s
- Current ATC_RAT_RLL_P: 0.09
- Current ATC_RAT_RLL_I: 0.05
- Current ATC_RAT_RLL_D: 0.004
- Vibrations: 1.5g

Flight Conditions: Cinematic drone, outdoor, calm""",
            "output": """Analysis:
- Roll performance is excellent (3.1° error, 2% overshoot)
- Settling time is optimal (0.8s)
- Vibration levels are good (1.5g)

Recommendations:
- No parameter changes needed
- Current PID values are well-tuned for this use case"""
        },
        {
            "instruction": "Analyze this flight log and recommend PID tuning adjustments.",
            "input": """Flight Log Analysis:
- Max Roll Error: 22.5°
- Overshoot: 18%
- Settling Time: 4.2s
- Current ATC_RAT_RLL_P: 0.15
- Current ATC_RAT_RLL_I: 0.03
- Current ATC_RAT_RLL_D: 0.001
- Vibrations: 3.8g

Flight Conditions: Heavy lift drone, outdoor, moderate wind""",
            "output": """Analysis:
- Significant overshoot (18%) - P gain far too high
- Long settling time (4.2s) - needs more I term
- High vibrations (3.8g) - filter adjustment needed
- Heavy payload requires different tuning approach

Recommendations:
1. Reduce ATC_RAT_RLL_P: 0.15 → 0.08 (47% reduction)
   Rationale: Heavy drones need lower P for stability
2. Increase ATC_RAT_RLL_I: 0.03 → 0.06 (100% increase)
   Rationale: More I term for better tracking
3. Increase ATC_RAT_RLL_D: 0.001 → 0.003 (200% increase)
   Rationale: More damping for heavy payload
4. Reduce INS_GYRO_FILTER: 80 → 50 Hz
   Rationale: Reduce vibration impact on control"""
        },
        {
            "instruction": "Analyze this flight log and recommend navigation parameter adjustments.",
            "input": """Flight Log Analysis:
- GPS Glitches: 12 occurrences
- Position Drift: 5.2m
- Navigation Error: 3.8m
- Current EK3_GPS_GAIN: 1.0
- Current EK3_ABIAS_P: 0.01

Flight Conditions: Survey drone, outdoor, moderate GPS signal""",
            "output": """Analysis:
- 12 GPS glitches indicate interference or configuration issues
- Position drift (5.2m) is too high for survey work
- Navigation error exceeds acceptable threshold

Recommendations:
1. Increase EK3_GPS_GAIN: 1.0 → 0.7 (reduce GPS reliance)
   Rationale: GPS signal not reliable, trust IMU more
2. Adjust EK3_ABIAS_P: 0.01 → 0.005 (lower variance)
   Rationale: Reduce accelerometer bias drift
3. Consider adding GPS glitch filtering
   Check: COMPASS_ENABLE, GPS_CONFIG parameters"""
        },
        {
            "instruction": "Analyze this flight log and recommend motor output adjustments.",
            "input": """Flight Log Analysis:
- Motor Temperature: 78°C (max)
- Current MOT_SPIN_MIN: 0
- Current MOT_SPIN_MAX: 9500
- Current MOT_BAT_VOLT_MAX: 52.2V
- Flight Duration: 18 minutes
- Battery Voltage End: 44.1V

Flight Conditions: FPV racing, aggressive flying""",
            "output": """Analysis:
- Motor temperature (78°C) is approaching safe limit
- End battery voltage (44.1V) suggests capacity stress
- Aggressive flight profile requires motor protection

Recommendations:
1. Set MOT_SPIN_MIN: 0 → 50
   Rationale: Prevent motor stalls at low throttle
2. Consider reducing max throttle for temperature management
3. Recommend adding motor thermal protection
   Check: MOT_TORQUE_BOOST, MOT_THT_AVERAGE_MAX parameters"""
        },
        {
            "instruction": "Analyze this flight log and recommend filter adjustments.",
            "input": """Flight Log Analysis:
- Vibrations: 4.2g (high)
- FFT Analysis: Peak at 180Hz
- Current INS_GYRO_FILTER: 80Hz
- Current INS_HNTC2_ENABLE: 0 (disabled)

Flight Conditions: 5" FPV quad, aggressive flying""",
            "output": """Analysis:
- High vibrations (4.2g) causing control issues
- FFT shows motor frequency peak at 180Hz
- Harmonic notch filter is disabled

Recommendations:
1. Enable INS_HNTC2_ENABLE: 0 → 1
   Rationale: Enable harmonic notch for vibration reduction
2. Set INS_HNTC2_FREQ: 180Hz
   Rationale: Target motor frequency peak
3. Set INS_HNTC2_BW: 40Hz
   Rationale: Narrow notch to preserve control bandwidth
4. Increase INS_GYRO_FILTER: 80 → 100Hz
   Rationale: Compensate for notch filter phase delay"""
        },
        {
            "instruction": "Analyze this flight log and recommend attitude controller adjustments.",
            "input": """Flight Log Analysis:
- Max Pitch Error: 18.3°
- Overshoot: 14%
- Oscillations: Visible wobble after maneuvers
- Current ATC_RAT_PIT_P: 0.12
- Current ATC_RAT_PIT_I: 0.05
- Current ATC_RAT_PIT_D: 0.002

Flight Conditions: Camera drone, smooth flying required""",
            "output": """Analysis:
- Significant pitch error (18.3°) and overshoot (14%)
- Visible oscillations indicate under-damped response
- Camera drone needs smooth, not aggressive, response

Recommendations:
1. Reduce ATC_RAT_PIT_P: 0.12 → 0.08 (33% reduction)
   Rationale: Smoother response for camera work
2. Increase ATC_RAT_PIT_D: 0.002 → 0.006 (200% increase)
   Rationale: Critical for damping oscillations
3. Reduce ATC_ANG_PIT_P: 4.5 → 3.0
   Rationale: Lower angle controller gain for smoothness"""
        },
        {
            "instruction": "Analyze this flight log and recommend yaw tuning adjustments.",
            "input": """Flight Log Analysis:
- Yaw Oscillations: Persistent after commands
- Yaw Settling Time: 3.5s (too slow)
- Current ATC_RAT_YAW_P: 0.15
- Current ATC_RAT_YAW_I: 0.05
- Current ATC_RAT_YAW_D: 0.0

Flight Conditions: FPV racer, fast maneuvers""",
            "output": """Analysis:
- Yaw oscillations persist - insufficient damping
- Slow settling time (3.5s) affects maneuverability
- D term is zero - missing critical damping component

Recommendations:
1. Add ATC_RAT_YAW_D: 0.0 → 0.003
   Rationale: D term essential for yaw damping
2. Increase ATC_RAT_YAW_I: 0.05 → 0.08 (60% increase)
   Rationale: Faster yaw correction
3. Increase ATC_RAT_YAW_MAX: 3600 → 5400
   Rationale: Allow faster yaw rate for racing"""
        }
    ]


def format_examples(examples: List[Dict]) -> str:
    """
    Format examples into training text.
    """
    formatted = []
    
    for ex in examples:
        instruction = ex.get("instruction", "")
        input_text = ex.get("input", "")
        output_text = ex.get("output", "")
        
        text = f"""Instruction: {instruction}

Input:
{input_text}

Output:
{output_text}"""
        
        formatted.append(text)
    
    return "\n\n".join(formatted)


def prepare_dataset(examples: List[Dict], tokenizer, max_length: int = 2048):
    """
    Prepare dataset for training.
    """
    formatted_texts = format_examples(examples)
    
    # Tokenize
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone()
    })
    
    return dataset


def train(
    model_name: str,
    data_dir: str,
    output_dir: str = "./ardupilot-tuner-model",
    rank: int = 16,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_steps: int = 4,
    seed: int = 42
):
    """
    Main training function.
    """
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"ArduPilot AI Tuner - QLoRA Fine-tuning")
    print(f"{'='*60}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none"
    )
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    print(f"\nLoading dataset from: {data_dir}")
    examples = load_dataset(data_dir)
    print(f"Loaded {len(examples)} training examples\n")
    
    dataset = prepare_dataset(examples, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal LM, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        fp16=True if device.type == "cuda" else False,
        dataloader_num_workers=0,
        report_to=None,
        remove_unused_columns=False,
    )
    
    # Train
    print("Starting training...\n")
    
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nTo use the fine-tuned model:")
    print(f"1. Convert to GGUF format for Ollama")
    print(f"2. Or use with transformers directly")
    print(f"\nModel saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for ArduPilot tuning")
    parser.add_argument("--model", type=str, default="qwen2.5:14b",
                        help="Base model name (HuggingFace or Ollama)")
    parser.add_argument("--data", type=str, default="../data/dataset/",
                        help="Training data directory")
    parser.add_argument("--output", type=str, default="./ardupilot-tuner-model",
                        help="Output directory for trained model")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (higher = more capacity, more memory)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_dir=args.data,
        output_dir=args.output,
        rank=args.rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()