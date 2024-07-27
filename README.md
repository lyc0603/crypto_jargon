# crypto_jargon
Project to study the cryptocurrency Chinese jargons.

[![python](https://img.shields.io/badge/Python-v3.11.7-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## Setup

```
git clone https://github.com/lyc0603/crypto_jargon.git
cd crypto_jargon
```

### Give execute permission to your script and then run `setup_repo.sh`

```
chmod +x setup_repo.sh
./setup_repo.sh
. venv/bin/activate
```

or follow the step-by-step instructions below between the two horizontal rules:

---

#### Create a python virtual environment

- MacOS / Linux

```bash
python3 -m venv venv
```

- Windows

```bash
python -m venv venv
```

#### Activate the virtual environment

- MacOS / Linux

```bash
. venv/bin/activate
```

- Windows (in Command Prompt, NOT Powershell)

```bash
venv\Scripts\activate.bat
```
#### Install toml

```
pip install toml
```

#### Install the project in editable mode

```bash
pip install -e ".[dev]"
```

#### Install pre-commit
```bash
pre-commit install
```

## Connect to a full node to fetch on-chain data

Connect to a full node using `ssh` with port forwarding flag `-L` on:

```zsh
ssh -L 8545:localhost:8545 satoshi.doc.ic.ac.uk
```

Assign URI value to `WEB3_PROVIDER_URI` in a new terminal:

```zsh
set -xg WEB3_PROVIDER_URI http://localhost:8545
```