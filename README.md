# ArduPilot AI Tuning System

A closed-loop AI system that iteratively optimizes ArduPilot flight parameters by analyzing flight logs. Upload your flight logs, get AI-powered parameter recommendations, apply them, fly again, and iterate until you reach optimal performance.

## âš¡ Quick Start (5 Minutes)

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/Asemdnn/ardupilot-autotune-.git
cd ardupilot-autotune-

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama (GPU Runtime)

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve &

# Pull the model (do this in a NEW terminal)
ollama pull qwen2.5:14b
```

### Step 3: Run the App

```bash
# Start the web server
python app/main.py
```

### Step 4: Use It

1. Open http://localhost:8000 in your browser
2. Upload a `.bin` or `.log` flight log
3. Get AI parameter recommendations
4. Apply to your drone, fly, upload next log
5. Repeat until parameters converge!

---

## ðŸ“‹ Complete Step-by-Step Guide

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA with 12GB+ VRAM (24GB recommended for 70B models) |
| **OS** | Linux, macOS, or Windows with WSL2 |
| **Python** | 3.10 or higher |
| **Disk Space** | 30GB+ for models and data |

### Phase 1: Environment Setup

#### 1.1 Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3-pip git curl

# For GPU support (CUDA)
nvidia-smi  # Verify GPU is detected
```

**macOS:**
```bash
brew install python3 git curl
```

**Windows:**
- Install Python 3.10+ from python.org
- Install Git from git-scm.com
- Use WSL2 for best compatibility

#### 1.2 Clone the Repository

```bash
git clone https://github.com/Asemdnn/ardupilot-autotune-.git
cd ardupilot-autotune-
```

#### 1.3 Install Python Packages

```bash
pip install -r requirements.txt
```

#### 1.4 Install Ollama & Download Model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama in background
ollama serve &

# Pull the base model (takes ~10-20 minutes first time)
# Option A: 14B model (requires 12GB VRAM)
ollama pull qwen2.5:14b

# Option B: 7B model (requires 8GB VRAM, faster)
ollama pull qwen2.5:7b
```

#### 1.5 Verify GPU Memory (Optional Check)

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

### Phase 2: Fine-Tune the Model (Optional but Recommended)

Fine-tuning creates a specialized "ArduPilot Expert" model. If you skip this, the base model will work but with less specialized knowledge.

#### 2.1 Prepare Training Data

The starter dataset is already in `data/dataset/training_data.json`. To improve your model:

1. Add more examples following the format in that file
2. Include edge cases: different frame types, flight modes, conditions
3. Aim for 50-200 examples for good results

#### 2.2 Run Fine-Tuning

```bash
pip install -r requirements-training.txt
cd training
python train.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data ../data/dataset/ \
    --output ../models/ardupilot-tuner \
    --rank 16 \
    --epochs 3
```

**Training Time Estimates (RTX 5090):**

| Model | Epochs | Time |
|-------|--------|------|
| 7B | 3 | ~20 minutes |
| 14B | 3 | ~45 minutes |
| 72B | 3 | ~4 hours |

#### 2.3 Use Fine-Tuned Model

After training, convert to GGUF format for Ollama, or modify `app/analyzer.py` to load the fine-tuned model directly.

---

### Phase 3: Run the Application

#### 3.1 Start the Server

```bash
cd ardupilot-autotune-
python app/main.py
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 3.2 Access the Web Interface

Open your browser to: http://localhost:8000

#### 3.3 Upload a Flight Log

1. Click "Choose File"
2. Select your ArduPilot `.bin` or `.log` file
3. Optionally add context (e.g., "FPV racer", "aerial photography")
4. Click to analyze

#### 3.4 Apply Recommendations

1. **Select your AI Inference Model** from the dropdown (\qwen2.5:7b\, b\, or :b\).
2. The AI will process your log and show:
   - **Metrics**: Roll error, mathematically derived Step-Response Overshoot, Settling time, and vibrations. *(Calculated securely via pandas/numpy vectorization against raw pymavlink iterators for memory-safe handling of 500MB+ logs).*
   - **Recommendations**: Dynamic parameter adjustments generated in real-time by your designated **Ollama LLM**.

Copy the recommended values and apply them via:
- Mission Planner
- QGroundControl
- Direct serial connection

---

### Phase 4: Iteration Loop

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      ITERATION WORKFLOW                       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                               â”‚
â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”             â”‚
â”‚   â”‚  Fly    â”‚â”€â”€â”€â–¶â”‚ Download â”‚â”€â”€â”€â–¶â”‚   Upload    â”‚             â”‚
â”‚   â”‚  Drone  â”‚    â”‚   Log    â”‚    â”‚   to AI     â”‚             â”‚
â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜             â”‚
â”‚                                          â”‚                   â”‚
â”‚                                          â–¼                   â”‚
â”‚            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”           â”‚
â”‚            â”‚  AI Analyzes & Recommends           â”‚           â”‚
â”‚            â”‚  New Parameters                     â”‚           â”‚
â”‚            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜           â”‚
â”‚                                 â”‚                              â”‚
â”‚                                 â–¼                              â”‚
â”‚                         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                      â”‚
â”‚                         â”‚ Apply to FC  â”‚                      â”‚
â”‚                         â”‚ & Fly Again  â”‚                      â”‚
â”‚                         â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜                      â”‚
â”‚                                 â”‚                              â”‚
â”‚                                 â–¼                              â”‚
â”‚                         [CONVERGED?]                          â”‚
â”‚                             â”‚                                 â”‚
â”‚                    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”                        â”‚
â”‚                    â”‚                 â”‚                        â”‚
â”‚                   YES               NO                        â”‚
â”‚                    â”‚                 â”‚                        â”‚
â”‚                    â–¼                 â–¼                        â”‚
â”‚              [OPTIMAL!]        [NEXT ITERATION]              â”‚
â”‚                                                               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**Typical Convergence:**
- **First flight**: Baseline parameters, initial recommendations
- **Flights 2-5**: Significant improvements each time
- **Flights 6-10**: Fine-tuning, smaller changes
- **Flight 10+**: Parameters stabilize â†’ You've converged!

---

### Phase 5: Troubleshooting

#### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Out of Memory (OOM)

If you get CUDA OOM errors:
1. Reduce batch size in training: `--batch-size 2`
2. Use smaller model: Switch from 14B to 7B
3. Enable gradient checkpointing in train.py

#### Log Parsing Errors

The analyzer supports both binary (.bin) and text (.log) formats. If your log doesn't parse:
1. Check it's a valid ArduPilot log
2. Try Mission Planner to convert format
3. Check the log has data messages (ATT, RATE, GPS, etc.)

---

## ðŸ”§ Configuration

### Changing the Training Model

Set the model when launching training:

```bash
python training/train.py --model Qwen/Qwen2.5-7B-Instruct
```

The app also accepts Ollama-style aliases (`qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:72b`) and maps them to HuggingFace model IDs internally.

### Adding Custom Parameters

Edit `app/parameters.py` to add more ArduPilot parameters:

```python
"NEW_PARAM": {
    "name": "Parameter Name",
    "description": "What it does",
    "range": [min, max],
    "default": default_value,
    "unit": "unit_name"
}
```

---

## ðŸ“ Project Structure

```
ardupilot-autotune-/
â”œâ”€â”€ README.md                    # This file
â”œâ”€â”€ requirements.txt            # Python dependencies
|-- requirements-training.txt   # Training dependencies
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ main.py                  # FastAPI web server
â”‚   â”œâ”€â”€ analyzer.py              # Flight log parser
â”‚   â”œâ”€â”€ parameters.py             # ArduPilot parameter database
â”‚   â””â”€â”€ templates/
â”‚       â””â”€â”€ index.html            # Web UI
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ logs/                     # Uploaded flight logs
â”‚   â”œâ”€â”€ dataset/                  # Training data
â”‚   â”‚   â””â”€â”€ training_data.json   # Starter dataset
â”‚   â””â”€â”€ outputs/                  # Analysis results
â””â”€â”€ training/
    â”œâ”€â”€ train.py                  # QLoRA fine-tuning script
    â””â”€â”€ lora_config.json          # LoRA configuration
```

---

## ðŸš€ Recommended Workflow

### For Best Results:

1. **Start with base model** - Verify everything works
2. **Collect 50+ real flight logs** - Diverse conditions
3. **Create high-quality training data** - Expert-curated
4. **Fine-tune** - Even 1-3 epochs helps significantly
5. **Iterate** - Upload logs, apply recommendations, repeat
6. **Converge** - Stop when recommendations stabilize

### Safety Tips:

- Always review recommendations before applying
- Keep a backup of working parameters
- Test significant changes in a safe environment first
- Start with small changes (10-25%) before aggressive tuning

---

## Development Checks

Run local tests:

```bash
python -m pytest -q
```

CI runs the same test suite on each push/PR via `.github/workflows/tests.yml`.

---

## ðŸ“š Additional Resources

- [ArduPilot Parameter List](https://ardupilot.org/copter/docs/parameters.html)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ArduPilot Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)

---

## License

MIT

