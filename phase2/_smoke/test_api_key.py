"""Minimal API-key sanity test. Run from repo root:

    /home/dongha/miniforge3/envs/mlebench/bin/python phase2/_smoke/test_api_key.py

Loads .env, calls gemini-2.5-flash with a trivial prompt, prints result.
Independent of our Phase 2 code path.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

# Load .env
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

print("=" * 60)
print(f"Repo root: {REPO_ROOT}")
print(f"GOOGLE_GENAI_USE_VERTEXAI = {os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')!r}")
print(f"GENAI_API_KEY present     = {bool(os.environ.get('GENAI_API_KEY'))}")
print(f"GENAI_API_KEY length      = {len(os.environ.get('GENAI_API_KEY', ''))}")
print("=" * 60)

# Import the same path PatientSim uses
sys.path.insert(0, str(SRC_DIR))
from models import gemini_response

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "Reply with exactly the single word: OK"},
]

print("\nCalling gemini-2.5-flash...")
try:
    resp = gemini_response(messages, model="gemini-2.5-flash", temperature=0, seed=42)
    text = resp.text if hasattr(resp, "text") else str(resp)
    print(f"\n✅ API call succeeded. Response: {text!r}")
except Exception as e:
    print(f"\n❌ API call FAILED")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error msg : {e}")
    sys.exit(1)
