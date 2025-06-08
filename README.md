# lyrgen

Lyric generator based on MIDI and spectral analysis.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Usage

```bash
python examples/demo.py --midi input.mid --mode dorian
```

## Development

- Run tests: `pytest`  
- Lint: `black --check src tests` and `flake8 src tests`
