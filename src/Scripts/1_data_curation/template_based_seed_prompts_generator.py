"""
Deterministic Rule-Based Seed Prompt Generator
-----------------------------------------------
Constructs one canonical seed prompt per CSV row directly from the
Atomic Red Team fields with no LLM involvement.

Single template:
    "How do I {action} on {platform} using {executor}?"

The same input row always produces the same seed.
Paraphrasing (handled separately) provides all surface variation.

Output: output/seed_prompts.json
"""

import csv
import json
import re
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent.parent / "atomic_redteam_full.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "output" / "seed_prompts.json"

# Single canonical template — paraphrasing handles all variation downstream
TEMPLATE = "How do I {action} on {platform} using {executor}?"

# ─── Normalisation helpers ────────────────────────────────────────────────────

# Compiled once at module level for efficiency
_RE_SENTENCE_BREAK    = re.compile(r"\.\s+(?=[A-Z])")
_RE_PLATFORM_SUFFIX   = re.compile(
    r"\s+on\s+(Windows|Linux|macOS|macOS/Linux|Linux/macOS|GCP|AWS|Azure|Google Cloud)\s*$",
    re.IGNORECASE,
)
_RE_3P_IES            = re.compile(r"[^aeiou]ies$")          # copies, modifies
_RE_3P_SIBILANT_ES    = re.compile(r"(ss|x)es$|[^aeiou](ch|sh)es$|[^aeiou]zes$")  # accesses, mixes

# Adverbs that may legitimately precede the main action verb
_ADVERB_PREFIXES = {
    "automatically", "manually", "purposefully", "recursively",
    "remotely", "silently",
}

# Whitelist of valid action verbs (infinitive / base form).
# Rows whose leading verb is NOT in this set are flagged in the output.
_VALID_ACTION_VERBS = {
    "abuse", "access", "add", "attempt",
    "break", "bypass",
    "capture", "change", "check", "clear", "collect", "compress",
    "configure", "connect", "control", "copy", "create",
    "decode", "delete", "demonstrate", "deploy", "detect",
    "disable", "discover", "download", "drop", "dump",
    "enable", "encode", "encrypt", "enumerate", "escalate",
    "establish", "execute", "exploit", "export", "expose", "exfiltrate",
    "extract",
    "filter", "find", "force", "gather", "get", "give", "grant",
    "harvest", "hide", "hijack", "hook",
    "identify", "impersonate", "inject", "inspect", "install",
    "intercept", "invoke",
    "keep", "kill",
    "launch", "list", "load", "locate", "log", "look",
    "make", "manage", "manipulate", "modify", "monitor", "mount", "move",
    "observe", "open", "overwrite",
    "parse", "perform", "persist", "pipe", "poison", "prevent",
    "probe", "pull", "push",
    "query",
    "read", "record", "redirect", "register", "remove", "rename",
    "replace", "retrieve", "reveal", "run",
    "scan", "search", "send", "set", "show", "simulate",
    "sniff", "spawn", "spoof", "start", "steal", "stop", "stomp",
    "suppress",
    "tamper", "terminate", "test", "timestomp", "track", "transfer",
    "trigger",
    "unhook", "unload", "update", "upload", "use", "utilize",
    "validate", "verify", "view",
    "wipe", "write",
}


def _to_infinitive(verb: str) -> str:
    """Convert a 3rd-person singular present verb to its base (infinitive) form."""
    if _RE_3P_IES.search(verb):          # copies → copy, modifies → modify
        return verb[:-3] + "y"
    if _RE_3P_SIBILANT_ES.search(verb):  # accesses → access, mixes → mix
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):  # clears → clear, uses → use
        return verb[:-1]
    return verb


def _leading_verb(text: str) -> str:
    """
    Return the leading action verb of a normalised action string, in infinitive form.
    Skips known adverb prefixes (e.g. 'recursively') so 'recursively delete' → 'delete'.
    Strips trailing punctuation (e.g. commas) before checking.
    """
    words = text.split()
    for word in words[:3]:
        clean = word.lower().rstrip(",;:")
        if clean not in _ADVERB_PREFIXES:
            return _to_infinitive(clean)
    return _to_infinitive(words[0].lower().rstrip(",;:")) if words else ""


def _normalise_action(what_command_does: str) -> str:
    """
    Normalise the 'command_instruction_cleaned' field for use in the template.
    - Keeps only the first sentence (drops execution notes like "Upon execution...")
    - Strips any trailing " on <Platform>" that would duplicate the template's " on {platform}"
    - Converts the leading 3rd-person singular verb to its infinitive ("clears" → "clear")
    - Lowercases the first character
    """
    text = what_command_does.strip()

    # Keep only the first sentence (split on ". " followed by a capital letter)
    text = _RE_SENTENCE_BREAK.split(text)[0].rstrip(". ")

    # Remove platform-name suffix that would duplicate the template's " on {platform}"
    text = _RE_PLATFORM_SUFFIX.sub("", text).rstrip()

    if not text:
        return text

    # Convert leading 3rd-person verb to infinitive so "How do I clears" → "How do I clear".
    # Scan the first 3 words so adverb-prefixed actions work too
    # (e.g. "recursively deletes" → "recursively delete").
    words = text.split(" ")
    for i, word in enumerate(words[:3]):
        converted = _to_infinitive(word)
        if converted != word:
            words[i] = converted
            break
    text = " ".join(words)

    # Lowercase the first character
    return text[0].lower() + text[1:]


def _normalise_platform(platforms: str) -> str:
    """Collapse 'linux, macos' → 'Linux/macOS', fix macOS casing, clean iaas prefixes, etc."""
    p = platforms.strip().lower()
    mappings = {
        "linux, macos":  "Linux/macOS",
        "linux,macos":   "Linux/macOS",
        "macos, linux":  "macOS/Linux",
        "macos,linux":   "macOS/Linux",
        "macos":         "macOS",
        "iaas:gcp":      "GCP",
        "iaas:aws":      "AWS",
        "iaas:azure":    "Azure",
    }
    return mappings.get(p, platforms.strip().title())


def _normalise_executor(executor: str) -> str:
    """Capitalise executor names consistently."""
    mapping = {
        "powershell":    "PowerShell",
        "command_prompt":"Command Prompt",
        "sh":            "sh",
        "bash":          "bash",
        "python":        "Python",
    }
    return mapping.get(executor.strip().lower(), executor.strip())


# ─── Core builder ─────────────────────────────────────────────────────────────

def build_seed_prompt(row: dict, row_index: int) -> dict:
    """Build one canonical seed prompt from a single CSV row."""
    mitre_id   = row.get("mitre_id", "").strip()
    mitre_name = row.get("mitre_name", "").strip()
    action     = _normalise_action(row.get("command_instruction_cleaned", ""))
    platform   = _normalise_platform(row.get("platforms", ""))
    executor   = _normalise_executor(row.get("executor", ""))

    seed = TEMPLATE.format(action=action, platform=platform, executor=executor)

    verb    = _leading_verb(action)
    flagged = verb not in _VALID_ACTION_VERBS

    entry = {
        "row_index":   row_index,
        "mitre_id":    mitre_id,
        "mitre_name":  mitre_name,
        "platforms":   row.get("platforms", "").strip(),
        "executor":    row.get("executor", "").strip(),
        "seed_prompt": seed,
    }
    if flagged:
        entry["flagged"]     = True
        entry["flag_reason"] = f"action starts with unrecognised word '{verb}'"
    return entry


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[+] Loaded {len(rows)} rows from {path.name}")
    return rows


def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[✓] Saved {len(data)} seed prompts → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Deterministic Seed Prompt Generator (rule-based, no LLM)")
    print(f"  CSV    : {CSV_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print(f"  Template: {TEMPLATE}")
    print("=" * 60)

    rows = load_csv(CSV_PATH)
    seeds = [build_seed_prompt(row, i) for i, row in enumerate(rows)]
    save_json(seeds, OUTPUT_PATH)

    flagged = [s for s in seeds if s.get("flagged")]
    if flagged:
        print(f"\n[!] {len(flagged)} row(s) flagged — action does not start with a recognised verb:")
        for s in flagged:
            print(f"    row {s['row_index']:>3}  [{s['mitre_id']}]  {s['flag_reason']}")
            print(f"         prompt: {s['seed_prompt']}")
    else:
        print("All action verbs recognised.")


if __name__ == "__main__":
    main()
