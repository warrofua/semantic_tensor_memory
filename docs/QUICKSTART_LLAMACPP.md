# Quick Start: llama.cpp Integration

**Fast-track guide to using llama.cpp with Semantic Tensor Analysis**

---

## What's New?

STA now supports **llama.cpp** for local LLM inference - faster and more memory-efficient than Ollama!

### Benefits
- ✅ **Faster inference** - No server overhead
- ✅ **Lower memory** - Quantized GGUF models (4-8GB)
- ✅ **GPU support** - Offload layers to GPU for speed
- ✅ **No server needed** - Direct model loading

---

## Installation (5 minutes)

### Step 1: Update Dependencies

```bash
# In project directory - install all dependencies
pip install -e .

# Or install individually if needed:
pip install llama-cpp-python matplotlib seaborn scipy
```

**Note on tkinter (for file browser):**
- **macOS/Windows**: Usually pre-installed with Python
- **Linux**: May need to install separately:
  ```bash
  sudo apt-get install python3-tk     # Ubuntu/Debian
  sudo yum install python3-tkinter    # CentOS/RHEL
  ```
- **If unavailable**: File browser won't work, but you can still type paths manually

**Note on llama.cpp compilation:** On some systems, you may need to install with specific flags:
```bash
# For Apple Silicon (M1/M2/M3) with Metal acceleration
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# For NVIDIA GPUs with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# CPU only (default)
pip install llama-cpp-python
```

### Step 2: Download a Model

**Option A: Quick Start (Smallest Model)**
```bash
mkdir -p ~/models
cd ~/models

# Download Phi-3 Mini (2.3GB) - fastest, good quality
wget https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

**Option B: Best Quality**
```bash
# Download Mistral 7B (4.1GB) - excellent for analysis
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Option C: Latest Model**
```bash
# Download Llama 3 8B (4.7GB) - latest Meta model
wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### Step 3: Launch STA

```bash
streamlit run app.py
```

---

## Usage (2 minutes)

### Configure llama.cpp in the UI

1. **Upload your data** in the sidebar (CSV with session_id, date, text)

2. **Navigate to any tab** with AI Insights (e.g., "AI Insights" tab, "Explain" tab)

3. **Open LLM Configuration:**
   - Look for "⚙️ LLM Configuration" expander
   - Click to expand

4. **Select Backend:**
   - Choose "llama.cpp" from dropdown

5. **Select Your Model:**

   **Option A: Use File Browser (Recommended)**
   - Click the "📁 Browse" button
   - Navigate to your models folder
   - Select your GGUF file (e.g., `Phi-3-mini-4k-instruct-q4.gguf`)
   - The path will auto-populate

   **Option B: Type Path Manually**
   ```
   /Users/yourusername/models/Phi-3-mini-4k-instruct-q4.gguf
   ```
   (Replace with your actual path from Step 2)

6. **Configure Performance (Optional):**
   - **CPU Threads**: 4 (default) - increase if you have more cores
   - **GPU Layers**: 0 (CPU only) - increase to offload to GPU (e.g., 20-30 for M1/M2/M3)

7. **Test It:**
   - Click "🧠 Analyze Journey" or "💡 Get Insights"
   - Watch the streaming response appear!

---

## Recommended Settings

### For Apple Silicon (M1/M2/M3)

```
Backend: llama.cpp
Model: Phi-3-mini-4k-instruct-q4.gguf
CPU Threads: 4
GPU Layers: 25 (for Metal acceleration)
```

### For High-End Desktop (32GB+ RAM, NVIDIA GPU)

```
Backend: llama.cpp
Model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
CPU Threads: 8
GPU Layers: 35 (offload most layers to GPU)
```

### For Standard Laptop (16GB RAM, no GPU)

```
Backend: llama.cpp
Model: Phi-3-mini-4k-instruct-q4.gguf
CPU Threads: 4
GPU Layers: 0 (CPU only)
```

---

## Troubleshooting

### "Model file not found"

**Solution:** Check the full path to your GGUF file:
```bash
ls -lh ~/models/*.gguf
```
Copy the full path and paste it into the "Model Path" field.

---

### "llama-cpp-python not installed"

**Solution:** Install the package:
```bash
pip install llama-cpp-python
```

If that fails, try with verbose output:
```bash
pip install llama-cpp-python --verbose
```

---

### Slow inference (>5 seconds per token)

**Try these optimizations:**

1. **Use a smaller model:**
   - Phi-3 Mini (2.3GB) is fastest
   - Q4 quantization is good balance of speed/quality

2. **Increase CPU threads:**
   - Set to number of physical cores (not logical)
   - Check: `sysctl -n hw.physicalcpu` (macOS) or `lscpu` (Linux)

3. **Enable GPU acceleration:**
   - Apple Silicon: Set GPU Layers to 25-35
   - NVIDIA: Set GPU Layers to 30-40

4. **Reduce context window:**
   - Modify `n_ctx` in code (default: 4096)
   - Smaller = faster but less context

---

### Out of memory errors

**Solutions:**

1. **Use a smaller model:**
   - Phi-3 Mini only needs ~4GB RAM

2. **Reduce GPU layers:**
   - Lower or set to 0 (CPU only)

3. **Close other applications:**
   - Free up RAM before running

---

## Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| Phi-3 Mini | 2.3GB | ⚡⚡⚡ | ⭐⭐⭐ | Quick testing, laptops |
| Mistral 7B | 4.1GB | ⚡⚡ | ⭐⭐⭐⭐ | Best balance |
| Llama 3 8B | 4.7GB | ⚡⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| Qwen2 7B | 4.4GB | ⚡⚡ | ⭐⭐⭐⭐ | Detailed analysis |

---

## Alternative: Keep Using Ollama

If you prefer Ollama, it still works perfectly! Just select "Ollama" in the backend dropdown.

**Ollama is better if:**
- You want easy model management (`ollama pull model`)
- You already have Ollama set up
- You use models across multiple tools

**llama.cpp is better if:**
- You want faster inference
- You want lower memory usage
- You want GPU acceleration on Apple Silicon
- You don't want a background server

---

## Advanced Configuration

### Edit in Code (Optional)

For more control, edit `src/semantic_tensor_analysis/chat/analysis.py`:

```python
# Around line 260-285
n_ctx: int = 4096      # Context window size
n_threads: int = 4     # CPU threads
n_gpu_layers: int = 0  # GPU layers to offload
```

### Load Multiple Models

You can switch models on the fly:
1. Change the "Model Path" field
2. Click "🧠 Analyze Journey" again
3. STA will reload with the new model

---

## Next Steps

Once you have llama.cpp working:

1. **Experiment with models:**
   - Try different sizes for your use case
   - Compare quality vs speed

2. **Optimize performance:**
   - Tune GPU layers for your hardware
   - Find the sweet spot for threads

3. **Explore analysis:**
   - Try different datasets
   - Compare insights from different models

4. **Read the docs:**
   - `AGENTS.md` - Full project guide
   - `IMPROVEMENT_PLAN.md` - Future enhancements
   - `README.md` - Complete documentation

---

## Support

**Questions?**
- Check `README.md` for full LLM setup section
- Review `IMPROVEMENT_PLAN.md` for known issues
- Open an issue on GitHub

**Works great?**
- Share your model recommendations
- Contribute optimizations
- Help improve documentation

---

**Happy analyzing! 🚀**

---

*Last updated: 2025-01-19*
