# Git Setup Guide - Elliott Wave Bot

## Quick Start

### 1. Initialize Git Repository
```bash
# Navigate to project directory
cd "C:\Users\Emre Yılmaz\Desktop\projects\elliottBot"

# Initialize Git
git init

# Add all files (respecting .gitignore)
git add .

# First commit
git commit -m "Initial commit: Elliott Wave Trading Bot"
```

### 2. Connect to Remote Repository (Optional)
```bash
# Add remote repository (replace with your GitHub/GitLab URL)
git remote add origin https://github.com/yourusername/elliott-wave-bot.git

# Push to remote
git push -u origin main
```

## What's Included/Excluded

### ✅ Files Tracked by Git
- Source code (`src/`)
- Examples (`examples/`)
- Documentation (`*.md`)
- Configuration template (`config_template.yaml`)
- Requirements files (`requirements*.txt`)
- License and readme files

### ❌ Files Ignored by Git
- **Configuration**: `config.yaml` (contains sensitive data)
- **Generated Charts**: `*.html` files
- **Data Files**: `*.csv`, historical data
- **Cache/Logs**: `logs/`, `cache/`, `*.log`
- **Python Cache**: `__pycache__/`, `*.pyc`
- **IDE Files**: `.vscode/`, `.idea/`
- **Virtual Environments**: `venv/`, `env/`
- **Trading Data**: Portfolio files, trading history

## Configuration Setup

### 1. Copy Template
```bash
cp config_template.yaml config.yaml
```

### 2. Edit Your Config
- Open `config.yaml` in your editor
- Update API keys (if using crypto features)
- Adjust wave detection parameters
- Set your preferred visualization settings

## Common Git Commands

### Daily Workflow
```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to remote
git push
```

### Branch Management
```bash
# Create new feature branch
git checkout -b feature/new-indicator

# Switch between branches
git checkout main
git checkout feature/new-indicator

# Merge feature branch
git checkout main
git merge feature/new-indicator
```

### Viewing History
```bash
# View commit history
git log --oneline

# View changes
git diff

# View specific file history
git log --follow src/analysis/wave_detector.py
```

## Best Practices

### Commit Messages
Use clear, descriptive commit messages:
```bash
git commit -m "Add RSI indicator to technical analysis"
git commit -m "Fix Fibonacci retracement calculation"
git commit -m "Update visualization colors and themes"
```

### Before Committing
1. **Test Your Code**: Run `python test_installation.py`
2. **Check Status**: Run `git status` to see what's being committed
3. **Review Changes**: Use `git diff` to review your changes

### Security
- **Never commit API keys** - they're in `.gitignore`
- **Keep config.yaml private** - contains sensitive settings
- **Use environment variables** for production secrets

## File Structure for Git

```
elliottBot/
├── .gitignore              # What to ignore
├── .git/                   # Git metadata (auto-created)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── config_template.yaml    # Configuration template
├── src/                    # Source code (tracked)
├── examples/               # Example scripts (tracked)
├── tools/                  # Utility scripts (tracked)
├── config.yaml            # Your config (ignored)
├── logs/                   # Log files (ignored)
├── *.html                  # Generated charts (ignored)
└── data/                   # Data files (ignored)
```

## Troubleshooting

### Large Files
If you accidentally add large files:
```bash
# Remove from staging
git reset HEAD large_file.csv

# Remove from tracking but keep locally
git rm --cached large_file.csv
```

### Undo Last Commit
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### View Ignored Files
```bash
# See what files are being ignored
git status --ignored
```

## Next Steps

1. **Initialize Repository**: `git init`
2. **First Commit**: Add and commit all files
3. **Set Up Remote**: Connect to GitHub/GitLab
4. **Configure Settings**: Copy and edit config template
5. **Start Developing**: Create feature branches for new work

Your Elliott Wave Bot is now ready for version control! 🚀
