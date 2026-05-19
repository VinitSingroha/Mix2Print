"""
llama_api_mode.py  —  GNN-CB API-only baseline for Mix2Print
Provider : Together AI (free tier)
Model    : meta-llama/Llama-3.3-70B-Instruct-Turbo
Repairs  : up to 5
Output   : run_llama_api/
"""

import os, re, subprocess, json, textwrap
from datetime import datetime
from pathlib import Path
from together import Together

# ── Configuration ────────────────────────────────────────────────────────────
MODEL             = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
TEMPERATURE       = 0
TOP_P             = 0.99
MAX_OUTPUT_TOKENS = 8000
MAX_REPAIRS       = 5
OUTPUT_DIR        = Path("run_llama_api")
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert GNN engineer. You will be given a graph regression competition task.

Your task is to produce a single, self-contained Python script that:
1. Loads graph data from .npy files (adjacency matrix A and node features X per graph)
2. Trains a GNN for graph-level regression predicting 3 targets: pressure, temperature, speed
3. Evaluates using NMAE = (1/3) * [MAE_pressure/1496 + MAE_temperature/228 + MAE_speed/90]
4. Saves predictions to predictions.csv

CRITICAL RULES:
- This is GRAPH REGRESSION — predict one (pressure, temperature, speed) per graph
- Each graph is one bioink formulation stored as separate .npy files
- Allowed libraries: pandas, numpy, scipy, scikit-learn ONLY — NO torch, NO torch_geometric
- Implement message passing manually using numpy matrix operations: H = relu(A @ X @ W)
- Use at least 2 GNN layers with mean pooling to get graph-level embedding
- Script must run as: python solution.py from repo root
- Set seeds: numpy and random to 42
- Do NOT import torch or torch_geometric — they are not installed

Required output format for predictions.csv:
id,pressure,temperature,speed
340,150.5,25.0,5.0
341,800.0,155.0,1.2
...

You must respond in EXACTLY this format:

<plan>
5-10 bullet plan covering: data loading, GNN architecture, training, validation, submission
</plan>

<code>
Complete runnable Python script here — no markdown fences inside
</code>

If you receive an error traceback, return a revised <plan> and <code> that fixes the specific error.
"""

USER_PROMPT = """## Competition Task: Mix2Print — 3D Bioprinting Parameter Prediction

## Data location
All graph data is in data/public/
- data/public/train_graphs/graph_{id}_A.npy  — adjacency matrix shape (n x n)
- data/public/train_graphs/graph_{id}_X.npy  — node features shape (n x 31)
- data/public/train_graphs/graph_{id}_y.npy  — targets shape (3,) = [pressure, temperature, speed]
- data/public/test_graphs/graph_{id}_A.npy   — test adjacency
- data/public/test_graphs/graph_{id}_X.npy   — test node features
- data/public/test_nodes.csv                 — test graph IDs to predict
- data/public/train.csv                      — training graph IDs

## How to load graph IDs
Train IDs: read from data/public/train.csv (look for an 'id' or index column)
Test IDs : read from data/public/test_nodes.csv (column 'id')

## Graph structure
- Each graph = one bioink formulation
- Nodes = biomaterials (variable number per graph, typically 2-6)
- Node features: 30-dim one-hot material identity + 1-dim concentration = 31 features
- Edges: fully connected (A is all-ones matrix)
- Target: 3 continuous values per graph [pressure kPa, temperature C, speed mm/s]

## Evaluation metric
NMAE = (1/3) * [MAE_pressure/1496 + MAE_temperature/228 + MAE_speed/90]
Lower is better.

## Required output
File: predictions.csv
Format:
  id,pressure,temperature,speed
  340,150.5,25.0,5.0
  341,800.0,155.0,1.2
  ...

- Exactly 120 rows (one per test graph)
- id matches test_nodes.csv
- All predicted values must be positive

## Allowed libraries ONLY
pandas, numpy, scipy, scikit-learn
DO NOT use torch or torch_geometric.

## Suggested numpy GNN pattern
For each graph:
  H0 = X                              # shape (n x 31)
  D = degree matrix of A
  A_norm = D^-0.5 @ A @ D^-0.5       # normalized adjacency
  H1 = relu(A_norm @ H0 @ W1)        # shape (n x hidden)
  H2 = relu(A_norm @ H1 @ W2)        # shape (n x out)
  graph_embed = mean(H2, axis=0)      # shape (out,)

Then use graph_embed as input to a regression head (e.g. sklearn MLPRegressor or Ridge).

Produce <plan> then <code>.
"""


def call_llm(client: Together, messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content


def extract_code(raw: str) -> str | None:
    """Pull Python code from <code>...</code> tags."""
    m = re.search(r"<code>(.*?)</code>", raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        code = re.sub(r"^```python\n?", "", code)
        code = re.sub(r"^```\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        return code.strip()
    return None


def run_script(script_path: Path) -> tuple[bool, str]:
    """Execute script_path; return (success, log)."""
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True, timeout=3600
        )
        log = result.stdout + result.stderr
        success = result.returncode == 0 and Path("predictions.csv").exists()
        return success, log
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired: script exceeded 60 minutes."


def list_created_files() -> str:
    files = [str(p) for p in Path(".").glob("*.csv")]
    return "\n".join(files) if files else "none"


def repair_message(traceback: str) -> dict:
    content = textwrap.dedent(f"""
        The script you produced failed with this traceback:

        <traceback>
        {traceback}
        </traceback>

        The current state of any files you wrote:

        <state>
        {list_created_files()}
        </state>

        REMINDER:
        - Do NOT use torch or torch_geometric — not installed
        - Use only: pandas, numpy, scipy, scikit-learn
        - Implement GNN message passing with numpy: H = relu(A_norm @ H @ W)
        - This is GRAPH REGRESSION — one prediction per graph

        Please return a revised <plan> and <code> that fixes the specific error.
    """).strip()
    return {"role": "user", "content": content}


def main():
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TOGETHER_API_KEY not set.\n"
            "PowerShell: $env:TOGETHER_API_KEY='your_key_here'\n"
            "CMD       : set TOGETHER_API_KEY=your_key_here"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = Together(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT},
    ]

    start   = datetime.now()
    success = False
    attempt = 0

    for attempt in range(1, MAX_REPAIRS + 2):
        tag = f"attempt_{attempt}"
        print(f"\n[*] {tag} — calling Llama via Together AI ...")

        raw = call_llm(client, messages)
        (OUTPUT_DIR / f"{tag}_raw_response.txt").write_text(raw, encoding="utf-8")

        code = extract_code(raw)
        if not code:
            print(f"    No <code> block found. Asking again ...")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "Your response did not contain a <code>...</code> block. "
                "Please return your full solution inside <code>...</code> tags."})
            continue

        sol_path = OUTPUT_DIR / f"{tag}_solution.py"
        sol_path.write_text(code, encoding="utf-8")
        print(f"    Running {sol_path} ...")

        ok, log = run_script(sol_path)
        (OUTPUT_DIR / f"{tag}_execution_log.txt").write_text(log, encoding="utf-8")

        log_tail = "\n".join(log.strip().splitlines()[-10:])
        print(f"    Log tail:\n{log_tail}")

        if ok:
            print(f"\n    Success on attempt {attempt}!")
            success = True
            import shutil
            shutil.copy(sol_path, OUTPUT_DIR / "final_solution.py")
            shutil.copy("predictions.csv", OUTPUT_DIR / "submission.csv")
            break
        else:
            print(f"    Failed. Sending traceback to Llama for repair ...")
            if attempt <= MAX_REPAIRS:
                messages.append({"role": "assistant", "content": raw})
                messages.append(repair_message(log[-3000:]))
            else:
                print("[!] Max repairs reached.")

    elapsed = (datetime.now() - start).total_seconds()

    summary = {
        "competition"  : "Mix2Print",
        "model"        : MODEL,
        "provider"     : "Together AI",
        "success"      : success,
        "attempts_used": attempt,
        "wall_time_sec": round(elapsed, 1),
        "output_dir"   : str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"  Model    : {MODEL}")
    print(f"  Success  : {success}")
    print(f"  Attempts : {attempt}")
    print(f"  Time     : {elapsed:.0f}s")
    if success:
        print(f"  Submit   : {OUTPUT_DIR}/submission.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
