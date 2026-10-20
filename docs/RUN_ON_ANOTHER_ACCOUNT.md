# Running seeds on your own Kaggle account

Thank you for helping. This runs the second half of seed 45 and two more seeds (46 and 47) of the experiments
on **your** Kaggle account's free GPU. Your laptop only uploads the code, starts
the sessions and downloads the results; the training happens on Kaggle.

It takes about 24 GPU-hours (five sessions of 4–6 hours each, one after
another; the queue budgets 27 to be safe). Kaggle gives every account 30
GPU-hours a week, so it fits in one week if the account has not used its GPU
this week. Your laptop has to stay on and connected while the queue runs (sleep off),
but you can use it normally.

Nothing here asks for or stores anyone else's password or key. Your Kaggle key
stays in your own `~/.kaggle/kaggle.json`, which only the Kaggle CLI reads.

## 1. Kaggle account (once)

1. On kaggle.com, **verify your phone number** (Settings → Phone verification).
   Without it Kaggle refuses GPU and internet access.
2. Settings → **API → Create New Token**. This downloads `kaggle.json`. Put it at
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS / Linux: `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)
3. Note your Kaggle **username** (shown on your profile page).

## 2. Set up the project (once)

You need Python 3.11 or newer and Git. On Windows, run everything below in
**Git Bash**.

```bash
cd Proof-of-Forgetting
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt kaggle   # macOS/Linux: .venv/bin/python
```

On macOS/Linux, write `.venv/bin/python` wherever these steps say
`.venv/Scripts/python`.

Check that it works (replace `YOUR_USERNAME`):

```bash
export POF_KAGGLE_USER=YOUR_USERNAME
export PYTHONUTF8=1
.venv/Scripts/kaggle quota            # should list your GPU hours
.venv/Scripts/python -m pytest -q     # should end with "passed"
```

Do **not** edit any file in the project: the runs refuse code that differs
from the committed version, so every seed runs exactly the same code.

## 3. Start the queue

```bash
export POF_KAGGLE_USER=YOUR_USERNAME
export PYTHONUTF8=1
.venv/Scripts/python scripts/kaggle_queue.py --init 45:canary 46:clean 46:canary 47:clean 47:canary
while true; do .venv/Scripts/python scripts/kaggle_queue.py --step; [ $? -ge 10 ] && break; sleep 300; done
```

Leave that window open. It creates a private dataset `pof-code` and a private
notebook `proof-of-forgetting-run` on your account, runs one session at a time,
and downloads each one's results into `results/`. Progress is in
`results/kaggle_logs/queue.log`; you can also watch the running notebook on
kaggle.com under Your Work → Code.

If the window is closed or the laptop restarts, just run the two `export` lines
and the `while` line again (not `--init`): it carries on from where it stopped.
If it stops with "GPU-hours left", the weekly quota ran out; run it again after
Kaggle's weekly reset.

## 4. Send the results back

When `queue.log` ends with `queue finished: every job ran`, zip the whole
`results` folder and send it back. That's all.

If anything fails, send `results/kaggle_logs/queue.log` and the last file in
`results/logs/`.
