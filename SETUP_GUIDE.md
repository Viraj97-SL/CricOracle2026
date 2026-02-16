# CricOracle 2026 — Complete Setup Guide (Step-by-Step)

## Prerequisites Checklist

Before you begin, make sure you have these installed:

| Tool | Version | Check Command | Install Link |
|------|---------|---------------|-------------|
| Python | 3.11+ | `python --version` | https://www.python.org/downloads/ |
| Git | 2.40+ | `git --version` | https://git-scm.com/downloads |
| PyCharm | 2024.x | Open app → Help → About | https://www.jetbrains.com/pycharm/download/ |
| Docker Desktop | 4.x | `docker --version` | https://www.docker.com/products/docker-desktop/ |
| Node.js | 18+ (for frontend later) | `node --version` | https://nodejs.org/ |

---

## STEP 1: Create the Project Locally (Terminal)

Open your terminal (PowerShell on Windows, Terminal on Mac/Linux).

```bash
# 1.1 Navigate to where you keep projects
cd ~/Projects          # Mac/Linux
cd C:\Users\YourName\Projects   # Windows

# 1.2 Create project folder
mkdir CricOracle2026
cd CricOracle2026

# 1.3 Initialise Git
git init

# 1.4 Create Python virtual environment
python -m venv .venv

# 1.5 Activate the virtual environment
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# 1.6 Verify you're in the venv (should show .venv path)
which python       # Mac/Linux
where python       # Windows
```

**IMPORTANT:** Every time you open a new terminal, you must activate the venv again (Step 1.5).

---

## STEP 2: Install Dependencies

```bash
# 2.1 Upgrade pip first (avoids dependency resolution issues)
pip install --upgrade pip setuptools wheel

# 2.2 Install core ML dependencies
pip install pandas==2.2.0 numpy==1.26.4 scikit-learn==1.4.0

# 2.3 Install gradient boosting libraries
pip install xgboost==2.0.3 lightgbm==4.3.0 catboost==1.2.3

# 2.4 Install PyTorch (CPU version — sufficient for development)
# For CPU only (recommended to start):
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
# For CUDA 12.1 (if you have an NVIDIA GPU):
# pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 2.5 Install API framework
pip install fastapi==0.109.0 uvicorn==0.27.0 pydantic==2.6.0

# 2.6 Install feature engineering & tuning tools
pip install optuna==3.5.0 shap==0.44.1

# 2.7 Install visualisation
pip install matplotlib==3.8.2 seaborn==0.13.1 plotly==5.18.0

# 2.8 Install testing tools
pip install pytest==8.0.0 pytest-cov==4.1.0 httpx==0.26.0

# 2.9 Install utilities
pip install python-dotenv==1.0.0 loguru==0.7.2 joblib==1.3.2 requests==2.31.0

# 2.10 Install Jupyter (for exploration notebooks inside PyCharm)
pip install jupyter==1.0.0 ipykernel==6.29.0

# 2.11 Install optimisation library (for squad selector)
pip install deap==1.4.1

# 2.12 Install code quality tools
pip install ruff==0.2.0 mypy==1.8.0

# 2.13 Freeze dependencies
pip freeze > requirements.txt

# 2.14 Register Jupyter kernel (so PyCharm/Colab can find your venv)
python -m ipykernel install --user --name=cricoracle --display-name="CricOracle2026"
```

---

## STEP 3: Open in PyCharm

1. Open PyCharm
2. File → Open → Select the `CricOracle2026` folder
3. PyCharm will detect the `.venv` automatically
4. If not: File → Settings → Project → Python Interpreter → Add Interpreter → Existing → Select `.venv/bin/python`
5. Wait for indexing to complete (bottom progress bar)

---

## STEP 4: Copy Project Files

Copy ALL the files from this starter kit into the project folder.
The folder structure should look exactly like this:

```
CricOracle2026/
├── .venv/                    (already created)
├── .github/workflows/ci.yml
├── .gitignore
├── .env.example
├── README.md
├── SETUP_GUIDE.md
├── requirements.txt          (already created)
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── conftest.py               (root-level, makes src importable)
│
├── data/
│   ├── raw/                  ← PUT YOUR CSV HERE
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── external/
│   │   └── .gitkeep
│   └── interim/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── weather.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── player.py
│   │   ├── team.py
│   │   ├── venue.py
│   │   ├── match_context.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── win_predictor.py
│   │   ├── score_predictor.py
│   │   ├── score_lstm.py
│   │   ├── squad_optimiser.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── explainability.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── validators.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── __init__.py
│       ├── predict.py
│       └── squad.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_models.py
│
├── scripts/
│   ├── train.py
│   └── predict.py
│
├── models/                   ← Trained model artifacts go here
│   └── .gitkeep
│
└── notebooks/
    └── 01_EDA.ipynb          ← Your exploration notebooks
```

---

## STEP 5: Place Your Data

```bash
# Copy your Cricsheet CSV into the data/raw/ folder
cp /path/to/your/t20_ball_by_ball_v2.csv data/raw/

# If you have multiple files:
cp /path/to/your/*.csv data/raw/
```

---

## STEP 6: Create Your .env File

```bash
# Copy the example and fill in your keys
cp .env.example .env
```

Edit `.env` with your actual API keys (get free keys from the links in .env.example).

---

## STEP 7: Verify Everything Works

```bash
# 7.1 Make sure venv is active
source .venv/bin/activate   # Mac/Linux

# 7.2 Run the data loader test
python -c "from src.config import settings; print(f'Project: {settings.PROJECT_NAME}')"
# Expected output: "Project: CricOracle2026"

# 7.3 Run the test suite
pytest tests/ -v
# Should see tests discovered (some will skip until data is loaded)

# 7.4 Run the linter
ruff check src/
# Should show no errors (or minor warnings)

# 7.5 Start the API (just to verify it boots)
uvicorn api.main:app --reload
# Open browser: http://localhost:8000/docs
# You should see the Swagger UI — press Ctrl+C to stop
```

---

## STEP 8: First Git Commit

```bash
git add .
git commit -m "feat: initial project scaffolding with full ML pipeline structure"
```

---

## STEP 9: Connect to GitHub

```bash
# 9.1 Create a new repo on GitHub (do NOT initialise with README)
# Go to: https://github.com/new
# Name: CricOracle2026
# Visibility: Public
# Do NOT add README, .gitignore, or license (we already have them)

# 9.2 Connect and push
git remote add origin https://github.com/YOUR_USERNAME/CricOracle2026.git
git branch -M main
git push -u origin main
```

---

## STEP 10: PyCharm Run Configurations

Set up these run configurations in PyCharm for quick access:

### Run Config 1: "Run Tests"
- Run → Edit Configurations → + → Python tests → pytest
- Target: `tests/`
- Working directory: project root
- Python interpreter: `.venv`

### Run Config 2: "Run API"
- Run → Edit Configurations → + → Python
- Script: `uvicorn`
- Parameters: `api.main:app --reload --port 8000`
- Working directory: project root

### Run Config 3: "Train Models"
- Run → Edit Configurations → + → Python
- Script path: `scripts/train.py`
- Parameters: `--model all`
- Working directory: project root

---

## What to Do Next

After setup is complete, follow this order:

1. **Run `notebooks/01_EDA.ipynb`** — Familiarise yourself with the data through the new pipeline
2. **Build features**: Edit `src/features/player.py` → run `pytest tests/test_features.py`
3. **Train baseline model**: Run `python scripts/train.py --model win_baseline`
4. **Iterate**: Add features → retrain → evaluate → repeat

Each subsequent step is detailed in the source files themselves with TODO comments.
