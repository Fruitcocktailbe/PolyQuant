# Beginner's Guide: How to Setup PolyQuant on AWS Lightsail

Welcome! This guide is written specifically for people with **zero computer programming or technical knowledge**. We will walk you through every single step to get PolyQuant running on your own private server in the cloud.

If you don't know what a word means, don't worry! Just follow the steps exactly as written.

---

## Step 1: Getting Your Server (AWS Lightsail)

Think of a "server" as an invisible computer running in a warehouse that never turns off. We are going to rent a cheap server from Amazon (AWS) to run our trading program.

1. Go to [Amazon Web Services (AWS)](https://aws.amazon.com/) and create an account if you don't have one.
2. Search for **Lightsail** in the search bar and click it.
3. Click the orange **Create instance** button.
4. **Instance Location**: Pick the region closest to you.
5. **Instance Image**: 
   - Select **Linux/Unix**.
   - Select **OS Only**.
   - Choose **Ubuntu 22.04 LTS**. (This is the operating system, like Windows or Mac, but for servers).
6. **Instance Plan**: Choose the **$12 USD/month** plan (it gives you 2 GB of RAM and 2 vCPUs). 
7. **Name your instance**: You can name it `polyquant-server`.
8. Click **Create instance**.

Wait a minute or two for the status to change from "Pending" to "Running".

---

## Step 2: Connecting to Your New Server

Now we need to log into the computer you just rented. Amazon makes this very easy.

1. On your Lightsail dashboard, you will see your new `polyquant-server`.
2. Click the little **orange terminal icon** (it looks like a black square with a `>_` inside).
3. A black window will pop up. This is your **Terminal**. You control the server by typing commands into this black box instead of clicking with a mouse.

**IMPORTANT:** To paste text into this black window, you usually have to **Right-Click** and select "Paste", or press `Ctrl + Shift + V`.

Or via terminal:
ssh -L 18789:127.0.0.1:18789 -L 8000:127.0.0.1:8000 -L 5173:127.0.0.1:5173 ubuntu@54.75.125.170 -i "C:\Users\Jeroen\Desktop\LightsailDefaultKey-eu-west-1polymarket.pem"
CHANGE KEY NAME IF NECESSARY
---

## Step 3: Preparing the Server

Just like a new phone needs updates, our server needs updates and some basic software.

Copy the single line of code below, paste it into your black terminal window, and press **Enter**. Wait for it to finish (it might take a minute or ask you to press `Y` to continue - if it does, press `Y` and Enter).

**Update the server:**
```bash
sudo apt update && sudo apt upgrade -y
```

**Install required software (these are tools like Python and Git):**
*(Copy this whole block, paste it, and press Enter)*
```bash
sudo apt install -y build-essential git redis-server python3-pip python3-venv pkg-config libssl-dev
```

**Install Rust (a programming language we need):**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
*Note: When it asks you what to do (1, 2, or 3), just press **1** and then **Enter**.*

**Load Rust into the terminal:**
```bash
source $HOME/.cargo/env
```

**Install SCIP (a math tool we need):**
```bash
sudo apt install -y scip
```

**Create "Swap" Memory (This gives your cheap server extra breathing room so it doesn't crash):**
*(Copy all these lines at once, paste them, and press Enter)*
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Step 4: Downloading PolyQuant

Now we will download the actual PolyQuant code from the internet using a tool called `git`.

1. Paste this command and press Enter:
```bash
git clone https://github.com/yourusername/PolyQuant.git
```
2. Go "inside" the PolyQuant folder by typing:
```bash
cd PolyQuant
```

---

## Step 5: Setting Up Python

PolyQuant uses the Python language. We need to install its specific dependencies.

1. Create a "virtual room" for Python so it doesn't mess with the rest of the server:
```bash
python3 -m venv venv
```
2. Enter the virtual room:
```bash
source venv/bin/activate
```
*(You will see the word `(venv)` appear on the left side of your typing line. This means it worked!)*

3. Install the requirements:
```bash
pip install -r requirements.txt
pip install -e .
```
*(This step might take a few minutes. Let the text scroll by until it stops and gives you control again).*

---

## Step 6: Adding Your Secret API Keys

The program needs your API keys (your secret passwords for Polymarket, OpenAI, etc.) to work. 

1. Copy the example configuration file so we can edit it:
```bash
cp .env.example .env
```
2. Open the file in a simple text editor called `nano`:
```bash
nano .env
```
3. You are now inside the text editor. You can use your keyboard arrows (Up, Down, Left, Right) to move around.
4. Fill in your API keys (e.g., your Limitless key, OpenRouter key, etc.).
5. **How to Save and Exit:**
   - Press `Ctrl + O` (the letter O, not zero) to Save.
   - Press **Enter** to confirm the file name.
   - Press `Ctrl + X` to Exit the editor.

---

## Step 7: Generating the Market Map

Before the robot can trade, it needs to scan the market and create a map.

Run this command:
```bash
python -m polyquant.map_maker
```
*Note: If it asks to download a model, let it. This takes a moment. Once it says it has finished making constraints, it will stop.*

---

## Step 8: How to Run Everything (And Never Turn It Off)

If you just run the code and then close the black window, the program will die. To keep it running forever, we use a tool called **Tmux**.

Think of Tmux like "tabs" in a web browser, but for your terminal. It keeps things running in the background even when you turn your home computer off.

### 1. Start a Tmux Session
```bash
tmux new -s polyquant
```
*(You will see a green bar appear at the bottom of your screen. You are now inside Tmux!)*

### 2. Tab 0: Run the Database (Redis) and Map Maker
Every time you open a new tab, you need to go into the folder and activate Python.
```bash
cd ~/PolyQuant
source venv/bin/activate
redis-server --daemonize yes
```

### 3. Tab 1: Run the Execution Engine (Rust)
Let's make a new tab to run the next part of the program.
- **Press `Ctrl + B`, let go of both keys, then press the letter `C`**. 
*(This creates a new blank window. You should see `1:bash` and `0:bash` at the bottom).*

Paste this to start the Rust engine:
```bash
cd ~/PolyQuant/oms-sidecar
cargo run --release
```
*(This will compile. It takes several minutes on the first try. Just wait until it starts running).*

### 4. Tab 2: Run the AI Agent Swarm (Python)
Let's make another tab for the brain of the robot.
- **Press `Ctrl + B`, let go, then press `C`**.

Paste this:
```bash
cd ~/PolyQuant
source venv/bin/activate
python -m polyquant.main trade
```

### 5. Tab 3: Run the Web Dashboard
Let's make one final tab for the website screen so you can view it on your phone/computer.
- **Press `Ctrl + B`, let go, then press `C`**.

Paste this:
```bash
cd ~/PolyQuant/web
npm install
chmod +x node_modules/.bin/vite
npm run dev -- --host
```

### 6. How to Leave (Detach) SAFELY
Right now, everything is running. If you click the 'X' on the top right of the window, you might break it. 
To leave safely so it keeps running forever:

- **Press `Ctrl + B`, let go, then press the letter `D`**.
*(It will say "[detached]". You can now close the black browser window! Your robot is trading 24/7!)*

---

## Step 9: Reconnecting Later!

If you wake up the next day and want to check on your robot:

1. Go to AWS Lightsail and open that Orange Terminal window again.
2. Type this to go back into your Tmux tabs:
```bash
tmux attach
```
3. You are back! To switch between your tabs (to see the different parts of the code):
   - **Press `Ctrl + B`, let go, then press the number `0`** (to see Tab 0)
   - **Press `Ctrl + B`, let go, then press the number `1`** (to see Tab 1)
   - **Press `Ctrl + B`, let go, then press the number `2`** (to see Tab 2)
   - **Press `Ctrl + B`, let go, then press the number `3`** (to see Tab 3)
4. When you are done looking, ALWAYS press:
   - **`Ctrl + B`, let go, then `D`** to detach safely again.

---

## Step 10: Viewing the Dashboard on your Phone/Computer

To view the pretty dashboard instead of looking at code:

1. Go to your AWS Lightsail dashboard. 
2. Click on the name of your server (`polyquant-server`).
3. Click on the **Networking** tab.
4. Scroll down to **IPv4 Firewall**. 
5. You need to "Open Ports". Click **+ Add rule**.
   - Protocol: TCP
   - Port range: `8000`
   - Click Create/Save.
6. Click **+ Add rule** again.
   - Protocol: TCP
   - Port range: `5173`
   - Click Create/Save.
7. Look for your server's **Public IP address** near the top right of the page (it looks like a bunch of numbers: `12.34.56.78`).
8. Open your web browser on your phone or laptop and type: `http://YOUR-PUBLIC-IP-ADDRESS:5173` 
   *(Example: `http://12.34.56.78:5173`)*

**Congratulations! You have officially set up PolyQuant!**
