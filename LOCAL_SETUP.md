# 🚀 PolyQuant 2.0: Local Setup Guide (Windows)

This document is designed to get you from "zero to trading" on your local Windows machine. 

---

## 🏁 Phase 0: System Prerequisites

Before we touch any code, you need two main tools installed on your Windows system.

### 1. Redis (The Database Cache)
PolyQuant uses Redis to store market prices in real-time.
*   **How to install**: Open PowerShell and run:
    `winget install Redis.Redis`
*   **Important**: After it installs, **Close your terminal and open a new one**.
*   **Verification**: Type `redis-cli ping`. You should see **`PONG`**.

### 2. SCIP (The Optimization Solver)
PolyQuant uses SCIP to solve complex arbitrage math in milliseconds.
*   **Verification**: If you already installed the SCIP Optimization Suite, you are good to go. We will verify the Python link in the next step.

---

## 🛠️ Phase 1: Python Environment Setup

We use a "Virtual Environment" (venv) to keep PolyQuant's libraries separate from the rest of your computer.

### 1. Create the Environment
Open a terminal in the `PolyQuant` folder and run:
```powershell
python -m venv venv
```
*(This creates a folder named `venv` in your project.)*

### 2. Activate the Environment
You must do this **every time** you open a new terminal to work on the bot.
```powershell
.\venv\Scripts\Activate.ps1
```
> [!TIP]
> **If you get a red "Scripts restricted" error**, run this once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> Then try the activation command again.

### 3. Install All Dependencies
This command reads the project files and installs everything (including `pyscipopt`, `openai`, `web3`, etc.):
```powershell
pip install -e .
```
*(The `-e` means "editable" – if we change a file, the bot sees it immediately.)*

---

## ⚙️ Phase 2: Configuration

You need to tell the bot your API keys.

1.  Find the file named `.env.example` in the root folder.
2.  **Duplicate it** and rename the copy to exactly `.env`.
3.  Open `.env` in a text editor (Notepad, VS Code, etc.).
4.  Fill in your keys (at minimum: `ALCHEMY_API_KEY`, `OPENROUTER_API_KEY`, and your `POLYGON_PRIVATE_KEY`).
5.  **Save the file.**

---

## 🧠 Phase 3: The Market Map (Step 1)

Before trading, the bot needs to "understand" the markets. This uses LLMs to find connections.

1.  **Terminal**: Ensure your `venv` is active (`(venv)` should show in green on the left).
2.  **Run logic**: 
    ```powershell
    python -m polyquant.main map --limit 50
    ```
3.  **What happens?**: The bot scans Polymarket, finds 50 liquid markets, and builds "Constraint Manifests" in the `.polyquant/` folder.
4.  **How often?**: Run this once to start, and then maybe once a week or whenever new big events (like elections or sports) happen.

---

## 💹 Phase 4: Ready to Trade (Step 2)

This is the main "Fast Brain" that watches prices and clicks "Buy/Sell".

1.  **Terminal**: Keep your environment active.
2.  **Command**:
    ```powershell
    python -m polyquant.main trade --mode paper
    ```
3.  **Monitor the UI**: While the bot is running, open your web browser to:
    **`http://localhost:8000`**
    (The dashboard is built-in! To enable it, you must first build the UI:
    ```powershell
    cd web
    npm install
    npm run build
    cd ..
    ```
    )

---

## ❓ FAQ: Do I need multiple terminals?

*   **Normally, NO**: One terminal can run the `trade` command, and that command automatically starts the Web UI server on port 8000.
*   **However**: If you want to run a "Map Scan" *while* the trader is running, you would open a **second terminal**, activate the `venv` again, and run the `map` command there.

---

## 🚨 Troubleshooting "Dummy Proof" Checklist

1.  **"ModuleNotFoundError: No module named..."**: Run `pip install -e .` again.
2.  **"redis-cli: command not found"**: Restart your computer or terminal. Redis was probably just installed.
3.  **"import pyscipopt failed"**: Ensure `SCIP_HOME` is in your Windows Environment Variables pointing to your SCIP folder.
4.  **"KeyboardInterrupt"**: This just means you pressed `Ctrl+C` to stop the bot. It's normal!
