# Fluence

**Adaptive text translation sample for high-load systems**

---

## 🚀 Startup

### 1. System requirements

* Nvidia GPU **compatible with CUDA 13.0**
* At least **16 GB of VRAM**
  *(You can reduce VRAM usage with custom settings.)*

> **Windows users:** Run the following in **PowerShell as Administrator**:

```powershell
# Allow scripts to run and enable TLS 1.2/1.3
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072

# Install Chocolatey
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install make
choco install make
make --version
```

---

### 2. Environment setup

For fast testing, **example files** with environment variables and settings are provided.

> **⚠ Warning:** Do **not** use `.example` files in production or for real tasks.

Copy `.example` files to working files:

```bash
make init-env
```

**You have to create your own protected environment files instead of using the examples for real tasks.**

---

### 3. Run the project

Start the service with:

```bash
make run
```

---

### 🔹 Notes

* Ensure all dependencies are installed before running.
* Adjust `.env` or JSON files if you need custom configuration.

---
