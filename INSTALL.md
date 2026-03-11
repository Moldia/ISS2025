# ISS2025 Environment Setup

This repository contains several modules that require separate Conda environments.  
A setup script is provided to automatically create these environments and install the packages.

---

## Requirements

Before running the setup script, make sure you have:

- Conda or Miniconda installed
- Git installed
- Internet access to download dependencies

---

## Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/yourlab/ISS2025.git
cd ISS2025
```

Run the setup script:

```bash
bash setup_envs.sh
```

---

## What the script does

The script automatically:

1. Creates Conda environments from the provided `.yml` files
2. Activates each environment
3. Installs the corresponding Python package using:

```bash
python setup.py install
```

4. Registers the environment as a **Jupyter kernel**
5. Deactivates the environment and moves to the next module

---

## Environments created

| Module | Environment Name | YAML File |
|------|------|------|
| Preprocessing | ISS_preprocessing | preprocessing.yml |
| Decoding | ISS_decoding | decoding.yml |
| Postprocessing | ISS_postprocessing | postprocessing.yml |
| CARE | ISS_CARE | ISS_CARE.yml |

---

## Using the environments in Jupyter

After installation, the environments will appear as kernels in Jupyter:

```
ISS_preprocessing
ISS_decoding
ISS_postprocessing
ISS_CARE
```

Select the appropriate kernel when running notebooks.

---

## Troubleshooting

If Conda is not initialized for your shell, run:

```bash
conda init
```

Then restart your terminal and run the setup script again.

