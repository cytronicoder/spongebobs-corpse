### Installation

```bash
git clone https://github.com/cytronicoder/spongebobs-corpse.git
cd spongebobs-corpse
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Developer checks:

```bash
pip install -e .[dev]
ruff check .
pytest
```
