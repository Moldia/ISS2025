# ISS2025 Environment Setup

This repository contains several modules that require separate Conda environments.  
A setup script is provided to automatically create or update these environments and install the packages.

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
git clone https://github.com/Moldia/ISS2025.git
cd ISS2025
```

Run the setup script:

```bash
bash setup_envs.sh
```

---

## Cloning the repository into a folder with another name

You can clone the repository into a folder with a different name by specifying the target directory in the `git clone` command.

```bash
git clone <repository_url> <new_folder_name>
cd <new_folder_name>
```

This will create a folder called `<new_folder_name>` instead of the default repository name.

You can then run the installer as usual:

```bash
bash setup_envs.sh
```

---

## Updating the pipeline

If the repository has been updated and you want to **update your local installation**, first pull the latest changes from GitHub and then run the installer again.

```bash
git pull
bash setup_envs.sh
```

This will:

- download the newest pipeline code
- update the conda environments if needed
- reinstall the Python packages so the latest code is used

The installer is safe to run multiple times.

---

## Repair / reinstall environments

If something in the environments becomes corrupted, you can remove the environments and reinstall everything.

Remove the environments:

```bash
conda remove -n ISS_preprocessing --all
conda remove -n ISS_decoding --all
conda remove -n ISS_postprocessing --all
conda remove -n ISS_CARE --all
```

Then reinstall them:

```bash
bash setup_envs.sh
```

---

## What the script does

The script automatically:

1. Creates Conda environments from the provided `.yml` files (or updates them if they already exist)
2. Activates each environment
3. Installs the corresponding Python package using:

```bash
pip install .
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