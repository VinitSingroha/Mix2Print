"""
Render leaderboard from CSV to Markdown.
"""

import csv
import json
import shutil
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "leaderboard" / "leaderboard.csv"
MD_PATH = ROOT / "leaderboard" / "leaderboard.md"
DOCS_CSV_PATH = ROOT / "docs" / "leaderboard.csv"
DATA_JS_PATH = ROOT / "docs" / "leaderboard_data.js"


def read_rows():
    """Read leaderboard CSV."""
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (r.get("team") or "").strip()]
    return rows


def main():
    rows = read_rows()
    
    # Sort by combined_nmae ascending (lower is better), then timestamp desc
    def score_key(r):
        try:
            return float(r.get("combined_nmae", "inf"))
        except:
            return float("inf")
    
    def ts_key(r):
        try:
            return datetime.fromisoformat(r.get("timestamp_utc", "").replace("Z", "+00:00"))
        except:
            return datetime.fromtimestamp(0)
    
    rows.sort(key=lambda r: (score_key(r), -ts_key(r).timestamp()))
    
    lines = []
    lines.append("# 🏆 Bioink GNN Challenge Leaderboard\n\n")
    lines.append("This leaderboard is **auto-updated** when a submission PR is scored. ")
    lines.append("For interactive search and filters, enable GitHub Pages and open **/docs/leaderboard.html**.\n\n")
    lines.append("**Metric:** Normalized MAE (NMAE) - Lower is better\n\n")
    
    lines.append("| Rank | Team | Model Type | NMAE | NMAE % | Date (UTC) | Notes |\n")
    lines.append("|---:|---|---|---:|---:|---|---|\n")
    
    for i, r in enumerate(rows, start=1):
        team = (r.get("team") or "").strip()
        model_type = (r.get("model_type") or "").strip()
        nmae = (r.get("combined_nmae") or "").strip()
        
        # Format NMAE as percentage
        try:
            nmae_pct = f"{float(nmae)*100:.2f}%"
        except:
            nmae_pct = "N/A"
        
        ts = (r.get("timestamp_utc") or "").strip()
        notes = (r.get("notes") or "").strip()
        
        # Badge for model type
        model_disp = f"`{model_type}`" if model_type else ""
        
        lines.append(f"| {i} | {team} | {model_disp} | {nmae} | {nmae_pct} | {ts} | {notes} |\n")
    
    if not rows:
        lines.append("\n*No submissions yet. Be the first!*\n")
    
    MD_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] Leaderboard rendered: {MD_PATH}")
    
    # Copy CSV to docs/ for interactive leaderboard
    shutil.copy2(CSV_PATH, DOCS_CSV_PATH)
    print(f"[OK] Leaderboard CSV copied to docs/: {DOCS_CSV_PATH}")

    # Write JS data for local offline viewing (bypass CORS)
    js_content = f"window.LEADERBOARD_DATA = {json.dumps(rows, indent=2)};"
    DATA_JS_PATH.write_text(js_content, encoding="utf-8")
    print(f"[OK] Leaderboard data.js generated: {DATA_JS_PATH}")


if __name__ == "__main__":
    main()
