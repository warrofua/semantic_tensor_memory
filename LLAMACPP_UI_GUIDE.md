# llama.cpp UI Guide - File Browser Feature

**Quick reference for using the new file browser in the llama.cpp configuration**

---

## New Feature: File Browser Button 📁

The llama.cpp configuration now includes a convenient file browser button to easily select your GGUF model files.

---

## How It Works

### Step-by-Step

1. **Open LLM Configuration**
   ```
   Expand the "⚙️ LLM Configuration" section
   ```

2. **Select Backend**
   ```
   Choose "llama.cpp" from the dropdown
   ```

3. **Use File Browser**
   ```
   Click the "📁 Browse" button
   ↓
   A file dialog opens
   ↓
   Navigate to your models folder
   ↓
   Select your .gguf file
   ↓
   Path auto-populates in the text field
   ```

4. **Confirm Selection**
   ```
   Status indicator shows:
   🤖 llama.cpp: model-name.gguf
   📊 Size: 4.1GB | ✅ Ready
   ```

---

## UI Layout

```
⚙️ LLM Configuration (Expandable)
│
├── LLM Backend: [llama.cpp ▼]
│
├── Model Selection:
│   ├── [📁 Browse]  [________________________]
│   │                     Model Path (GGUF)
│   │
│   └── Filters: *.gguf, *.*
│       Initial directory: ~/models (if exists) or ~/
│
├── Performance Settings:
│   ├── CPU Threads: [4 ▼]
│   └── GPU Layers: [0 ▼]
│
└── ☑ Show recommended models
```

---

## Status Indicators

### When No Model Selected
```
🤖 llama.cpp: ⚠️ No model selected
```

### When Model Selected & Valid
```
🤖 llama.cpp: mistral-7b-instruct-v0.2.Q4_K_M.gguf
📊 Size: 4.1GB | ✅ Ready
```

### When Path Invalid
```
🤖 llama.cpp: ⚠️ No model selected
(Path doesn't exist)
```

---

## File Browser Features

### Filters
- **GGUF Models** (*.gguf) - Recommended
- **All Files** (*.*) - If you need other formats

### Starting Directory
- First checks: `~/models/`
- Falls back to: `~/` (home directory)

### File Selection
- Single file selection only
- Shows file name in title bar
- Updates immediately on selection

---

## Manual Path Entry (Alternative)

If file browser doesn't work or you prefer manual entry:

1. **Type or paste full path:**
   ```
   /Users/yourname/models/model.gguf
   ```

2. **Path validation:**
   - Checks if file exists
   - Shows warning if invalid
   - Updates status when valid

---

## Examples

### Example 1: macOS with Downloaded Model
```
Path: /Users/josh/models/Phi-3-mini-4k-instruct-q4.gguf
Status: 🤖 llama.cpp: Phi-3-mini-4k-instruct-q4.gguf
        📊 Size: 2.3GB | ✅ Ready
```

### Example 2: Custom Location
```
Path: /Volumes/External/LLM-Models/mistral-7b-v0.2.gguf
Status: 🤖 llama.cpp: mistral-7b-v0.2.gguf
        📊 Size: 4.1GB | ✅ Ready
```

### Example 3: Downloads Folder
```
Click Browse → Navigate to ~/Downloads/
Select: Llama-3-8B-Instruct-Q4_K_M.gguf
Path auto-fills: /Users/josh/Downloads/Llama-3-8B-Instruct-Q4_K_M.gguf
```

---

## Troubleshooting

### "File browser requires tkinter"
**Cause:** tkinter not installed or available

**Solution:**
```bash
# macOS - usually pre-installed with Python
# If not, reinstall Python from python.org

# Linux - install tkinter
sudo apt-get install python3-tk  # Ubuntu/Debian
sudo yum install python3-tkinter  # CentOS/RHEL

# Then use manual path entry as fallback
```

---

### "Error opening file browser"
**Cause:** Display/GUI not available (headless server, SSH without X11)

**Solution:**
Use manual path entry:
```
Type full path: /path/to/your/model.gguf
```

---

### Browse button doesn't work
**Fallback Options:**
1. Type path manually in text field
2. Copy path from Finder/Explorer:
   - Right-click file
   - "Copy Path" or "Get Info"
   - Paste into text field

---

## Tips & Tricks

### Tip 1: Organize Your Models
```bash
# Create a dedicated models folder
mkdir -p ~/models

# Download models there
cd ~/models
wget https://huggingface.co/.../model.gguf

# Browser will remember this location!
```

### Tip 2: Symlinks for Easy Access
```bash
# Create symlink in home directory
ln -s /Volumes/External/Models ~/models

# Now browser starts at ~/models
# But files are on external drive
```

### Tip 3: Recent Paths
```
The path you select is saved in session state
It will persist during your Streamlit session
Reopen config to see your last selection
```

### Tip 4: Model Info at a Glance
```
Status shows file size (in GB)
Quick way to confirm you selected the right model:
- Phi-3 Mini: ~2.3GB
- Mistral 7B: ~4.1GB
- Llama 3 8B: ~4.7GB
```

---

## Benefits of File Browser

✅ **No typing errors** - Select file visually
✅ **No path confusion** - See folder structure
✅ **Quick selection** - One click vs typing full path
✅ **File filters** - Only show .gguf files
✅ **Visual confirmation** - See file name and size
✅ **Cross-platform** - Works on macOS, Linux, Windows

---

## Comparison: Browse vs Manual Entry

| Feature | Browse Button | Manual Entry |
|---------|--------------|--------------|
| Speed | ⚡⚡⚡ Fast | ⚡ Slower |
| Accuracy | ✅ High (no typos) | ⚠️ Can have typos |
| Visual | ✅ See files/folders | ❌ Just text |
| Filter | ✅ .gguf filter | ❌ No filter |
| Requires | tkinter | Nothing |
| Works with | GUI displays | All environments |

**Recommendation:** Use Browse when available, Manual Entry as fallback

---

## Future Enhancements (Planned)

- 🔮 Model metadata preview (size, quantization level)
- 🔮 Recent models dropdown
- 🔮 Drag-and-drop support
- 🔮 Model validation (check if valid GGUF)
- 🔮 Quick model download from Hugging Face

---

## Related Documentation

- **QUICKSTART_LLAMACPP.md** - Full setup guide
- **README.md** - LLM backend section
- **IMPROVEMENT_PLAN.md** - Future enhancements

---

**Last Updated:** 2025-01-19
**Feature Added:** File browser with tkinter
**Status:** ✅ Production ready
