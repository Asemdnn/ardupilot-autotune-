# ArduPilot AI Tuning System

A closed-loop AI system that iteratively optimizes ArduPilot flight parameters by analyzing flight logs. Upload your flight logs, get AI-powered parameter recommendations, apply them, fly again, and iterate until you reach optimal performance.

## ⚡ Quick Start (5 Minutes)

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

## 📋 Complete Step-by-Step Guide

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
cd ardupilot-ai-tuner
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
cd training
python train.py \
    --model qwen2.5:14b \
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
cd ardupilot-ai-tuner
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

The AI will show:
- **Metrics**: Roll error, overshoot, settling time, vibrations
- **Recommendations**: Specific parameter changes with rationale

Copy the recommended values and apply them via:
- Mission Planner
- QGroundControl
- Direct serial connection

---

### Phase 4: Iteration Loop

```
┌──────────────────────────────────────────────────────────────┐
│                      ITERATION WORKFLOW                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────┐    ┌─────────┐    ┌─────────────┐             │
│   │  Fly    │───▶│ Download │───▶│   Upload    │             │
│   │  Drone  │    │   Log    │    │   to AI     │             │
│   └─────────┘    └─────────┘    └──────┬──────┘             │
│                                          │                   │
│                                          ▼                   │
│            ┌─────────────────────────────────────┐           │
│            │  AI Analyzes & Recommends           │           │
│            │  New Parameters                     │           │
│            └──────────────────┬──────────────────┘           │
│                                 │                              │
│                                 ▼                              │
│                         ┌───────────────┐                      │
│                         │ Apply to FC  │                      │
│                         │ & Fly Again  │                      │
│                         └───────┬───────┘                      │
│                                 │                              │
│                                 ▼                              │
│                         [CONVERGED?]                          │
│                             │                                 │
│                    ┌────────┴────────┐                        │
│                    │                 │                        │
│                   YES               NO                        │
│                    │                 │                        │
│                    ▼                 ▼                        │
│              [OPTIMAL!]        [NEXT ITERATION]              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Typical Convergence:**
- **First flight**: Baseline parameters, initial recommendations
- **Flights 2-5**: Significant improvements each time
- **Flights 6-10**: Fine-tuning, smaller changes
- **Flight 10+**: Parameters stabilize → You've converged!

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

## 🔧 Configuration

### Changing the Model

Edit `app/main.py` to change the model:

```python
# For 7B model (faster, less VRAM)
model_name = "qwen2.5:7b"

# For 14B model (better reasoning)
model_name = "qwen2.5:14b"

# For 70B model (best, needs 24GB VRAM)
model_name = "qwen2.5:70b"
```

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

## 📁 Project Structure

```
ardupilot-ai-tuner/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── app/
│   ├── main.py                  # FastAPI web server
│   ├── analyzer.py              # Flight log parser
│   ├── parameters.py             # ArduPilot parameter database
│   └── templates/
│       └── index.html            # Web UI
├── data/
│   ├── logs/                     # Uploaded flight logs
│   ├── dataset/                  # Training data
│   │   └── training_data.json   # Starter dataset
│   └── outputs/                  # Analysis results
└── training/
    ├── train.py                  # QLoRA fine-tuning script
    └── lora_config.json          # LoRA configuration
```

---

## 🚀 Recommended Workflow

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

## 📚 Additional Resources

- [ArduPilot Parameter List](https://ardupilot.org/copter/docs/parameters.html)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [ArduPilot Tuning Guide](https://ardupilot.org/copter/docs/tuning.html)

---

## License

MIT
