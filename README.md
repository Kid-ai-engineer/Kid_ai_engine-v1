🤖 AI Workshop - Complete Installation Guide

📋 Overview

AI Workshop is a beginner-friendly AI training system that allows you to train machine learning models without writing complex code. Perfect for students, educators, and hobbyists!

🚀 Quick Installation

For Everyone (Simplest Method)

1. Download the engine:
   ```bash
   # Save the Python code as AI_WORKSHOP_ENGINE.py
   
   ```
2. Run the setup:
   ```bash
   python AI_WORKSHOP_ENGINE.py --setup
   ```
3. Follow the on-screen instructions
   · Creates workspace at ~/AI_Workshop/
   · Adds ai command to your system
   · Sets up all necessary directories

Manual Installation

If the automatic setup doesn't work, follow these steps:

---

🖥️ System Requirements

Minimum:

· Python 3.8 or higher
· 4GB RAM
· 2GB free disk space
· Internet connection (for downloading models)

Recommended:

· Python 3.9+
· 8GB RAM
· NVIDIA GPU with CUDA support (optional, for faster training)
· 10GB free disk space

---

📦 Step-by-Step Installation

Step 1: Install Python Dependencies

Open terminal/command prompt and run:

```bash
# For Windows:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas pillow requests scikit-learn

# For macOS/Linux:
pip3 install torch torchvision torchaudio
pip3 install numpy pandas pillow requests scikit-learn

# For GPU support (NVIDIA only):
# Visit: https://pytorch.org/get-started/locally/
```

Step 2: Set Up the AI Engine

1. Save the engine file:
   ```bash
   # Create a folder for AI Workshop
   mkdir ~/AI_Workshop
   cd ~/AI_Workshop
   
   # Save the Python code as ai_engine.py
   # (Copy the entire code into ai_engine.py)
   ```
2. Make it executable (Mac/Linux):
   ```bash
   chmod +x ai_engine.py
   ```

Step 3: Configure GitHub Repository

IMPORTANT: Before using, you MUST update the GitHub URL:

1. Open ai_engine.py in a text editor
2. Find this line (around line 36):
   ```python
   GITHUB_REPO = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/main"
   ```
3. Replace with your GitHub repository URL:
   ```python
   GITHUB_REPO = "https://raw.githubusercontent.com/yourusername/your-repo/main"
   ```

Step 4: Create Workspace Directories

The engine will create these automatically, but you can create them manually:

```bash
# Windows:
mkdir %USERPROFILE%\AI_Workshop
mkdir %USERPROFILE%\AI_Workshop\models
mkdir %USERPROFILE%\AI_Workshop\projects
mkdir %USERPROFILE%\AI_Workshop\data
mkdir %USERPROFILE%\AI_Workshop\outputs

# Mac/Linux:
mkdir -p ~/AI_Workshop/{models,projects,data,outputs}
```

Step 5: Create Command Aliases

For Windows:

1. Create ai.bat in your home directory:
   ```batch
   @echo off
   cd "%USERPROFILE%\AI_Workshop"
   python ai_engine.py %*
   ```
2. Add to PATH (Optional):
   · Move ai.bat to C:\Windows\System32\
   · Or add its location to your system PATH

For Mac/Linux:

1. Create ai script in ~/AI_Workshop/:
   ```bash
   #!/bin/bash
   cd ~/AI_Workshop
   python3 ai_engine.py "$@"
   ```
2. Make it executable:
   ```bash
   chmod +x ~/AI_Workshop/ai
   ```
3. Add to PATH:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   echo 'export PATH="$HOME/AI_Workshop:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

---

🎯 Verification Test

After installation, verify everything works:

```bash
# Check workspace
ai config

# List models (should show "No models installed yet")
ai models

# Create a test project
ai new test_project
```

---

🔧 Troubleshooting

Common Issues and Solutions:

1. "python: command not found"

· Windows: Install Python from python.org
· Mac: brew install python
· Linux: sudo apt install python3 python3-pip

2. "torch not found"

```bash
# Try specific version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

3. "Permission denied" (Mac/Linux)

```bash
# Add execute permission
chmod +x ~/AI_Workshop/ai_engine.py
chmod +x ~/AI_Workshop/ai
```

4. "Module not found" errors

```bash
# Install missing packages
pip install [missing-package-name]
# Common ones:
pip install importlib.util  # Actually part of Python standard library
pip install pathlib        # Also part of standard library
```

5. Windows PATH issues

```powershell
# In PowerShell as Administrator:
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Users\YourName\AI_Workshop", "User")
```

---

📂 Directory Structure Explained

After installation, you'll have:

```
AI_Workshop/
│
├── ai_engine.py          # Main engine file
├── ai                    # Linux/Mac launcher
├── ai.bat                # Windows launcher
├── README.txt            # Quick start guide
│
├── models/               # Downloaded model blueprints
│   ├── CatDog.py
│   └── [Other models]
│
├── projects/             # Your AI projects
│   ├── cat_classifier/
│   │   ├── cat_classifier.ai  # Configuration file
│   │   └── data/              # Your training data
│   └── [Other projects]
│
├── data/                 # Shared datasets
│
└── outputs/              # Trained models
    ├── trained_model.pth
    └── [Other outputs]
```

---

🎮 Quick Start Tutorial

1. Create Your First AI Project

```bash
ai new cat_detector
```

2. Prepare Your Data

· Put cat images in: ~/AI_Workshop/projects/cat_detector/data/
· Or create CSV file with features and labels

3. Edit the AI File

Open ~/AI_Workshop/projects/cat_detector/cat_detector.ai:

```
import required
required{CatDog}
model.name: {
model.read(data/)
model.train{
[80%,20%]
seed: 42
}
model.save cat_detector_model
}

/start
```

4. Train Your Model

```bash
ai train cat_detector.ai
```

5. Test Your Model

```bash
ai test outputs/cat_detector_model.pth new_cat_image.jpg
```

---

🌐 Setting Up Your Own Model Repository

To share models or create custom ones:

1. Create a GitHub repository
2. Add model files in a models/ folder:
   ```
   your-repo/
   ├── models/
   │   ├── CatDog.py
   │   ├── NumberRecognizer.py
   │   └── SentimentAnalyzer.py
   └── README.md
   ```
3. Update the GITHUB_REPO in your ai_engine.py:
   ```python
   GITHUB_REPO = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main"
   ```
4. Share with others: They can use your models by setting the same URL

---

🔄 Updating the Engine

Manual Update:

1. Download the latest ai_engine.py
2. Replace your old file
3. Keep your models/, projects/, data/, outputs/ folders

Backup Your Work:

```bash
# Backup entire workspace
tar -czf ai_workshop_backup.tar.gz ~/AI_Workshop

# Backup just your projects
cp -r ~/AI_Workshop/projects ~/ai_projects_backup
```

---

📚 Learning Resources

For Beginners:

1. Start with images: Try classifying cats vs dogs
2. Try tabular data: Use CSV files with simple features
3. Experiment with splits: Try [70%,15%,15%] instead of [80%,20%]

Next Steps:

1. Create your own model blueprints
2. Share models with friends
3. Join the community (if applicable)

Sample Projects to Try:

· Image Classification: Cats vs Dogs, Flowers, Fruits
· Tabular Data: House price prediction, Student grades
· Custom Models: Modify existing blueprints

---

🆘 Getting Help

Check Logs:

```bash
# Run with verbose output
python ai_engine.py train your_project.ai --verbose
```

Common Error Messages:

Error Solution
"Model not found" Run ai install MODEL_NAME
"No data found" Check data path in .ai file
"Out of memory" Reduce batch size or image size
"Download failed" Check internet connection, update GITHUB_REPO

Debug Mode:

```python
# Add to ai_engine.py before main()
import traceback
traceback.print_exc()
```

---

🎉 Congratulations!

You've successfully installed the AI Workshop! Now you can:

✅ Train AI models with simple commands
✅ Classify images and data
✅ Create and share projects
✅ Learn AI concepts hands-on

Start creating: ai new my_first_ai

---

📄 License & Credits

This engine is designed for educational purposes. Built with:

· PyTorch
· NumPy/Pandas
· PIL/Pillow
· Requests

Remember: Always update the GITHUB_REPO variable before use!

