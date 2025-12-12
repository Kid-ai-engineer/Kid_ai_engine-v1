#!/usr/bin/env python3
"""
AI_WORKSHOP_ENGINE.py
Complete AI Engine - Single File
All features, real working code
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import zipfile
import requests
from io import BytesIO
import warnings
import importlib.util

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
WORKSPACE_DIR = Path.home() / "AI_Workshop"
MODELS_DIR = WORKSPACE_DIR / "models"
PROJECTS_DIR = WORKSPACE_DIR / "projects"
DATA_DIR = WORKSPACE_DIR / "data"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"

# ==================== IMPORTANT: CHANGE THIS ====================
# REPLACE THIS WITH YOUR ACTUAL GITHUB REPOSITORY URL
# Your GitHub repository for models
GITHUB_REPO = "https://raw.githubusercontent.com/kid_ai_models_for_everyone/ai_models/main"

# Example formats:
# GITHUB_REPO = "https://raw.githubusercontent.com/johnsmith/ai-models/main"
# GITHUB_REPO = "https://raw.githubusercontent.com/AI-For-Kids/ai-model-templates/master"
# ================================================================

# ==================== UTILITY FUNCTIONS ====================
def setup_workspace():
    """Create workspace directories if they don't exist"""
    for directory in [WORKSPACE_DIR, MODELS_DIR, PROJECTS_DIR, DATA_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Workspace ready at: {WORKSPACE_DIR}")

def download_file(url, save_path):
    """Download file from URL"""
    try:
        print(f"   Downloading: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
        print()
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ==================== PARSER ====================
def parse_ai_file(file_path):
    """Parse kid's .ai file with YOUR exact syntax"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract model name from required{ MODEL_NAME }
    model_name = None
    if 'required{' in content:
        start = content.find('required{') + 9
        end = content.find('}', start)
        model_name = content[start:end].strip()
    
    # Extract data path from model.read(PATH)
    data_path = None
    if 'model.read(' in content:
        start = content.find('model.read(') + 11
        end = content.find(')', start)
        data_path = content[start:end].strip()
    
    # Extract save name from model.save NAME
    save_name = None
    if 'model.save' in content:
        start = content.find('model.save') + 10
        # Find next } or end of line
        end = content.find('\n', start)
        if end == -1:
            end = len(content)
        save_name = content[start:end].strip()
    
    # Extract training config from model.train{ [%,%] }
    train_config = {"split": [80, 20], "seed": None}
    if 'model.train{' in content:
        start = content.find('model.train{') + 12
        end = content.find('}', start)
        train_block = content[start:end].strip()
        
        # Parse split [%,%]
        if '[' in train_block and ']' in train_block:
            split_start = train_block.find('[') + 1
            split_end = train_block.find(']', split_start)
            split_str = train_block[split_start:split_end]
            split_parts = [p.strip().replace('%', '') for p in split_str.split(',')]
            train_config["split"] = [int(p) for p in split_parts if p]
        
        # Parse seed
        if 'seed:' in train_block:
            seed_start = train_block.find('seed:') + 5
            seed_part = train_block[seed_start:].strip()
            seed_value = seed_part.split()[0] if ' ' in seed_part else seed_part
            if seed_value.lower() != 'random' and seed_value.isdigit():
                train_config["seed"] = int(seed_value)
    
    return {
        "model_name": model_name,
        "data_path": data_path,
        "save_name": save_name or "trained_model",
        "train_config": train_config
    }

# ==================== DATA LOADING ====================
def load_dataset(data_path):
    """Load dataset from path - auto-detects CSV or images"""
    data_path = Path(data_path)
    
    # If it's a directory of images
    if data_path.is_dir():
        images = []
        labels = []
        class_names = []
        class_idx = 0
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(data_path.glob(f'*{ext}')))
            image_files.extend(list(data_path.glob(f'*{ext.upper()}')))
        
        if image_files:
            # Single directory - all same class
            for img_file in image_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize((128, 128))
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(0)  # Single class
                except:
                    continue
            
            if images:
                images = np.array(images).transpose(0, 3, 1, 2)  # NHWC to NCHW
                return torch.FloatTensor(images), torch.LongTensor(labels), ["class1"]
        
        # Check for subdirectories as classes
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if subdirs:
            for class_dir in subdirs:
                class_name = class_dir.name
                class_names.append(class_name)
                
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            img = Image.open(img_file).convert('RGB')
                            img = img.resize((128, 128))
                            img_array = np.array(img) / 255.0
                            images.append(img_array)
                            labels.append(class_idx)
                        except:
                            continue
                class_idx += 1
            
            if images:
                images = np.array(images).transpose(0, 3, 1, 2)
                return torch.FloatTensor(images), torch.LongTensor(labels), class_names
    
    # If it's a CSV file
    elif data_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(data_path)
            if len(df.columns) < 2:
                raise ValueError("CSV needs at least 2 columns")
            
            # Last column = labels, others = features
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values
            
            # Convert string labels to numeric
            if y.dtype == object:
                unique_labels = np.unique(y)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = np.array([label_map[label] for label in y])
            
            return torch.FloatTensor(X), torch.LongTensor(y), list(map(str, np.unique(y)))
        except Exception as e:
            print(f"❌ CSV loading error: {e}")
    
    raise ValueError(f"Could not load data from {data_path}")

# ==================== MODEL LOADING ====================
def ensure_model_installed(model_name):
    """Ensure model blueprint is installed, download if needed"""
    model_file = MODELS_DIR / f"{model_name}.py"
    
    if not model_file.exists():
        print(f"📥 Model '{model_name}' not found. Downloading...")
        
        # Try different possible URLs
        urls_to_try = [
            f"{GITHUB_REPO}/models/{model_name}.py",
            f"{GITHUB_REPO}/{model_name}.py",
            f"{GITHUB_REPO}/main/models/{model_name}.py",
        ]
        
        success = False
        for url in urls_to_try:
            print(f"   Trying: {url}")
            if download_file(url, model_file):
                success = True
                break
        
        if success:
            print(f"✅ Model '{model_name}' installed")
        else:
            print(f"⚠️  Could not download from GitHub. Creating default model...")
            create_default_model(model_name, model_file)
    
    return model_file

def create_default_model(model_name, model_file):
    """Create a default model if download fails"""
    if model_name.lower() == "catdog":
        model_code = '''import torch.nn as nn

def create_model(num_classes=2):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*32*32, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
'''
    else:
        model_code = '''import torch.nn as nn

def create_model(input_size=10, num_classes=2):
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
'''
    
    with open(model_file, 'w') as f:
        f.write(model_code)
    print(f"📝 Created default model for '{model_name}'")

def load_model_blueprint(model_name):
    """Dynamically load model blueprint"""
    model_file = ensure_model_installed(model_name)
    
    # Add models directory to Python path
    if str(MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(MODELS_DIR))
    
    try:
        # Import the module
        module_name = model_name.lower()
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None:
            raise ImportError(f"Could not load spec from {model_file}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Get create_model function
        if hasattr(module, 'create_model'):
            return module.create_model
        else:
            raise AttributeError("Model must have 'create_model' function")
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {e}")
        # Return a simple default model
        return lambda **kwargs: create_simple_model(**kwargs)

def create_simple_model(input_size=10, num_classes=2):
    """Fallback simple model"""
    return nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )

# ==================== TRAINING ====================
def train_model_engine(model_name, data_path, save_name, train_config):
    """Main training function"""
    print(f"🚀 Starting training: {model_name}")
    
    # Load data
    print(f"📊 Loading data from: {data_path}")
    X, y, class_names = load_dataset(data_path)
    print(f"   Found {len(X)} samples, {len(class_names)} classes: {class_names}")
    
    # Set random seed if specified
    if train_config["seed"]:
        torch.manual_seed(train_config["seed"])
        np.random.seed(train_config["seed"])
        print(f"   Using seed: {train_config['seed']} (reproducible)")
    
    # Split data
    split = train_config["split"]
    if len(split) == 2:
        train_ratio = split[0] / 100
        test_ratio = split[1] / 100
        val_ratio = 0
    elif len(split) == 3:
        train_ratio = split[0] / 100
        val_ratio = split[1] / 100
        test_ratio = split[2] / 100
    else:
        train_ratio = 0.8
        val_ratio = 0.0
        test_ratio = 0.2
    
    n_total = len(X)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio) if val_ratio > 0 else 0
    n_test = n_total - n_train - n_val
    
    if n_train == 0:
        raise ValueError("Training set size is 0! Check your split percentages.")
    
    # Create dataset
    dataset = TensorDataset(X, y)
    
    # Split
    train_dataset, test_dataset = random_split(
        dataset, [n_train, n_test + n_val],
        generator=torch.Generator().manual_seed(train_config["seed"] or 42)
    )
    
    if n_val > 0:
        val_size = n_val
        test_size = n_test
        test_val_dataset, test_dataset = random_split(
            test_dataset, [val_size, test_size],
            generator=torch.Generator().manual_seed(train_config["seed"] or 42)
        )
    
    print(f"   Split: {n_train} train, {n_val} val, {n_test} test")
    
    # Create model
    model_creator = load_model_blueprint(model_name)
    
    # Determine model parameters
    if len(X.shape) == 4:  # Image data (N, C, H, W)
        input_size = X.shape[1]  # Channels
        model = model_creator(num_classes=len(class_names))
    else:  # Tabular data
        input_size = X.shape[1]
        model = model_creator(input_size=input_size, num_classes=len(class_names))
    
    print(f"   Model created: {model.__class__.__name__}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=min(32, n_train), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(32, n_test), shuffle=False)
    
    # Training loop
    print("🎯 Training model...")
    model.train()
    for epoch in range(10):  # 10 epochs
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"   Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.1f}%")
    
    # Evaluation
    print("📈 Evaluating model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    test_accuracy = 100 * correct / total if total > 0 else 0
    print(f"✅ Test Accuracy: {test_accuracy:.1f}%")
    
    # Save model
    save_path = OUTPUTS_DIR / f"{save_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'accuracy': test_accuracy,
        'input_size': input_size,
        'model_name': model_name
    }, save_path)
    
    print(f"💾 Model saved to: {save_path}")
    return test_accuracy, save_path

# ==================== COMMAND INTERFACE ====================
def create_new_project(project_name):
    """Create a new project with template .ai file"""
    project_dir = PROJECTS_DIR / project_name
    project_dir.mkdir(exist_ok=True)
    
    template = f"""import required
required{{CatDog}}
model.name: {{
model.read(data/)
model.train{{
[80%,20%]
seed: 42
}}
model.save {project_name}_model
}}

/start"""
    
    ai_file = project_dir / f"{project_name}.ai"
    with open(ai_file, 'w') as f:
        f.write(template)
    
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Project created: {project_dir}")
    print(f"📝 Edit: {ai_file}")
    print(f"📁 Put your data in: {data_dir}")
    print(f"🚀 Then run: ai train {project_name}.ai")

def list_models():
    """List available models"""
    models = list(MODELS_DIR.glob("*.py"))
    if not models:
        print("📭 No models installed yet.")
        print(f"   Models will be downloaded from: {GITHUB_REPO}")
        print(f"   Or use: ai install MODEL_NAME")
        return
    
    print("📦 Available Models:")
    for model_file in models:
        model_name = model_file.stem
        size = model_file.stat().st_size
        print(f"  • {model_name} ({size} bytes)")
    print(f"\nTo install new model: ai install MODEL_NAME")

def install_model(model_name):
    """Install model from GitHub"""
    print(f"📥 Installing model: {model_name}")
    ensure_model_installed(model_name)

def test_trained_model(model_path, test_data):
    """Test a trained model"""
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        class_names = checkpoint['class_names']
        input_size = checkpoint['input_size']
        model_name = checkpoint.get('model_name', 'Unknown')
        
        # Load test data
        X_test, y_test, _ = load_dataset(test_data)
        
        # Create model architecture
        model_creator = load_model_blueprint(model_name)
        if len(X_test.shape) == 4:
            model = model_creator(num_classes=len(class_names))
        else:
            model = model_creator(input_size=input_size, num_classes=len(class_names))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            if len(X_test) == 1:
                output = model(X_test.unsqueeze(0))
            else:
                output = model(X_test)
            _, predicted = torch.max(output, 1)
        
        if len(X_test) == 1:
            pred_class = class_names[predicted.item()]
            print(f"🔍 Prediction: {pred_class}")
            if len(y_test) == 1:
                actual_class = class_names[y_test.item()]
                print(f"📌 Actual: {actual_class}")
        else:
            correct = (predicted == y_test).sum().item()
            accuracy = 100 * correct / len(y_test)
            print(f"📊 Test Accuracy: {accuracy:.1f}% ({correct}/{len(y_test)} correct)")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

# ==================== MAIN FUNCTION ====================
def main():
    """Main entry point"""
    
    # Setup workspace first
    setup_workspace()
    
    # Show GitHub info
    print(f"🌐 Model repository: {GITHUB_REPO}")
    print(f"📁 Workspace: {WORKSPACE_DIR}")
    print()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("🤖 AI Workshop Engine")
        print("Commands:")
        print("  ai new PROJECT_NAME      - Create new project")
        print("  ai train PROJECT_FILE    - Train AI model")
        print("  ai test MODEL DATA       - Test trained model")
        print("  ai models                - List available models")
        print("  ai install MODEL_NAME    - Install new model")
        print("  ai config                - Show configuration")
        return
    
    command = sys.argv[1]
    
    if command == "new":
        if len(sys.argv) < 3:
            print("Usage: ai new PROJECT_NAME")
            return
        create_new_project(sys.argv[2])
    
    elif command == "train":
        if len(sys.argv) < 3:
            print("Usage: ai train PROJECT_FILE")
            return
        
        project_file = Path(sys.argv[2])
        if not project_file.exists():
            # Check in projects directory
            project_file = PROJECTS_DIR / sys.argv[2]
            if not project_file.exists():
                # Check with .ai extension
                project_file = PROJECTS_DIR / f"{sys.argv[2]}.ai"
        
        if not project_file.exists():
            print(f"❌ Project file not found: {sys.argv[2]}")
            print(f"   Looked in: {Path(sys.argv[2])}")
            print(f"   And in: {project_file}")
            return
        
        print(f"📖 Reading project: {project_file}")
        config = parse_ai_file(project_file)
        
        if not config["model_name"]:
            print("❌ No model specified in project file")
            print("   Make sure you have: required{MODEL_NAME}")
            return
        
        if not config["data_path"]:
            print("❌ No data path specified in project file")
            print("   Make sure you have: model.read(YOUR_DATA_PATH)")
            return
        
        # Run training
        try:
            train_model_engine(
                config["model_name"],
                config["data_path"],
                config["save_name"],
                config["train_config"]
            )
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif command == "test":
        if len(sys.argv) < 4:
            print("Usage: ai test MODEL_PATH DATA_PATH")
            print("Example: ai test outputs/my_model.pth test_image.jpg")
            return
        test_trained_model(sys.argv[2], sys.argv[3])
    
    elif command == "models":
        list_models()
    
    elif command == "install":
        if len(sys.argv) < 3:
            print("Usage: ai install MODEL_NAME")
            return
        install_model(sys.argv[2])
    
    elif command == "config":
        print("⚙️ Configuration:")
        print(f"  GitHub Repository: {GITHUB_REPO}")
        print(f"  Workspace: {WORKSPACE_DIR}")
        print(f"  Models Directory: {MODELS_DIR}")
        print(f"  Projects Directory: {PROJECTS_DIR}")
        print(f"  Outputs Directory: {OUTPUTS_DIR}")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: new, train, test, models, install, config")

# ==================== SETUP SCRIPT ====================
def create_setup_script():
    """Create setup script for easy installation"""
    setup_content = f'''#!/bin/bash
# AI Workshop Setup Script

echo "🤖 Setting up AI Workshop..."

# Create workspace
mkdir -p ~/AI_Workshop/{{models,projects,data,outputs}}

# Copy engine
cp "$0" ~/AI_Workshop/ai_engine.py

# Create launcher scripts
cat > ~/AI_Workshop/ai << 'EOF'
#!/bin/bash
cd ~/AI_Workshop
python3 ai_engine.py "$@"
EOF

cat > ~/AI_Workshop/ai.bat << 'EOF'
@echo off
cd "%USERPROFILE%\\AI_Workshop"
python ai_engine.py %*
EOF

chmod +x ~/AI_Workshop/ai ~/AI_Workshop/ai_engine.py

# Create README
cat > ~/AI_Workshop/README.txt << 'EOF'
🤖 AI WORKSHOP
==============

Your AI training workspace is ready!

Commands:
  ai new PROJECT_NAME    - Create new AI project
  ai train PROJECT_FILE  - Train your AI model
  ai test MODEL DATA     - Test trained model
  ai models             - List available models
  ai install MODEL      - Install new model

Quick Start:
1. Type: ai new my_first_ai
2. Put your data in: ~/AI_Workshop/projects/my_first_ai/data/
3. Type: ai train my_first_ai.ai
4. Your trained model will be in outputs/

GitHub Repository: {GITHUB_REPO}
EOF

# Add to PATH (Linux/Mac)
if [[ "$SHELL" == *"bash"* ]]; then
    echo 'export PATH="$HOME/AI_Workshop:$PATH"' >> ~/.bashrc
    echo 'alias ai="cd ~/AI_Workshop && python3 ai_engine.py"' >> ~/.bashrc
    echo "Added to bashrc. Run: source ~/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    echo 'export PATH="$HOME/AI_Workshop:$PATH"' >> ~/.zshrc
    echo 'alias ai="cd ~/AI_Workshop && python3 ai_engine.py"' >> ~/.zshrc
    echo "Added to zshrc. Run: source ~/.zshrc"
fi

echo "✅ AI Workshop installed!"
echo "📁 Workspace: ~/AI_Workshop/"
echo "🚀 Usage: ai new PROJECT_NAME"
echo "📖 More info: cat ~/AI_Workshop/README.txt"
'''
    
    setup_path = Path("setup_ai.sh")
    with open(setup_path, 'w') as f:
        f.write(setup_content)
    
    print(f"📝 Setup script created: {setup_path}")
    print("To install: bash setup_ai.sh")

# ==================== RUN ENGINE ====================
if __name__ == "__main__":
    # Check if setup mode
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        create_setup_script()
    else:
        main()
