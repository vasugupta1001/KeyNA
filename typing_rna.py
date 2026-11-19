"""
Typing DNA Gate - passwordless login using keystroke dynamics.

How it works
------------
1. ENROLL: the user types 10 fixed sentences. For every sample we record
   letter-pair (digraph) timings, dwell time, flight time, speed, error rate
   and key-rollover rate. Together they form the user's typing profile
   (a mean and standard deviation for each feature).
2. LOGIN: the server issues a NEW random sentence (never repeated for that
   user) with a one-time token. The user types it, and each feature is turned
   into a z-score:  |new value - profile mean| / profile std
   The z-scores are combined into one weighted score.
3. DECISION: score <= GRANT_THRESHOLD -> GRANTED
             score <= STEP_UP_THRESHOLD -> STEP_UP (ask for a second factor, e.g. OTP)
             otherwise                  -> DENIED

Thresholds and weights are STARTING VALUES. Tune them by testing with real
people (measure the false accept and false reject rates).
"""

import difflib
import json
import math
import random
import secrets
import sqlite3
import statistics
import time
from contextlib import closing

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
DB_PATH = "typing_dna.db"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PAUSE_MS = 1000              # gaps longer than this are thinking pauses, not rhythm
MAX_EVENTS = 5000            # reject absurdly large payloads
MIN_KEYSTROKE_COVERAGE = 0.8 # typed text must come from real key presses (blocks paste)
MIN_ACCURACY = 0.90          # typed text must match the sentence at least 90%
MIN_PAIR_COUNT = 5           # a letter pair needs 5+ enrollment samples to enter the profile
MIN_OVERLAP = 12             # a login needs 12+ letter pairs that exist in the profile
CHALLENGE_TTL_SEC = 300      # a sentence must be typed within 5 minutes
MAX_FAILED_ATTEMPTS = 3      # lock the account after 3 failed logins...
LOCKOUT_WINDOW_SEC = 900     # ...within 15 minutes
MAX_LOGIN_SAMPLES_IN_PROFILE = 10  # profile adapts using the last 10 successful logins
Z_CAP = 5.0                  # one wild value can't dominate the score

GRANT_THRESHOLD = 1.5
STEP_UP_THRESHOLD = 2.0
SHOW_SCORE_DETAILS = True    # handy for demos and tuning; set False in real use (it helps attackers)

# Feature weights (sum to 1.0). Rhythm features carry most of the weight.
WEIGHTS = {
    "digraph": 0.35,
    "dwell_mean": 0.20,
    "flight_mean": 0.20,
    "wpm": 0.10,
    "error_rate": 0.10,
    "rollover_rate": 0.05,
}
# Minimum standard deviation for each feature, so a very consistent
# enrollment can't produce a tiny std and divide-by-almost-zero.
STD_FLOOR = {
    "digraph": 10.0,       # ms
    "dwell_mean": 5.0,     # ms
    "flight_mean": 10.0,   # ms
    "wpm": 3.0,            # words per minute
    "error_rate": 1.0,     # backspaces per 100 characters
    "rollover_rate": 0.05, # fraction
}
SCALAR_FEATURES = ["dwell_mean", "flight_mean", "wpm", "error_rate", "rollover_rate"]

# ---------------------------------------------------------------------------
# Sentences
# Rules: lowercase letters and spaces only, common words, no names or famous
# quotes, 45-60 characters, rich in common English letter pairs.
# ---------------------------------------------------------------------------
TARGET_PAIRS = {"th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
                "es", "or", "te", "ed", "st", "ar", "is", "it", "ng", "ou"}

# Fixed enrollment set: each of the 20 target pairs appears at least 5 times across it.
ENROLL_SENTENCES = [
    "the teacher started reading the stories to the students",
    "she wanted to finish the painting before the evening",
    "there is a small garden near the station entrance",
    "our neighbour is sitting in the kitchen ironing clothes",
    "the driver turned around and waited at the corner",
    "he found an old letter under the stairs in the store",
    "the farmer carried the boxes into the house at noon",
    "it rained so hard that the river entered the streets",
    "the doctor listened to the patient and wrote messages",
    "we are visiting the city market on monday morning",
]

# Login sentences: [who] [did what] [to what] [where/when] -> 4,096 combinations.
WHO = ["the teacher", "the farmer", "my sister", "the doctor",
       "our neighbour", "the student", "the driver", "the painter"]
DID = ["painted", "carried", "noticed", "ordered",
       "repaired", "washed", "started", "returned"]
WHAT = ["the heavy table", "a green kettle", "the broken gate", "an old letter",
        "the station clock", "a winter coat", "the garden hose", "the iron lantern"]
WHERE = ["near the river", "in the evening", "after the rain", "at the market",
         "behind the house", "on the third floor", "under the bridge", "before dinner"]
MIN_TARGET_PAIRS_PER_LOGIN_SENTENCE = 12

_rng = random.SystemRandom()  # unpredictable, unlike a counter or the default RNG


def count_target_pairs(sentence: str) -> int:
    return sum(
        1 for word in sentence.split()
        for i in range(len(word) - 1)
        if word[i:i + 2] in TARGET_PAIRS
    )


def new_login_sentence(used: set[str]) -> str | None:
    """Random template sentence with enough target pairs that this user has never seen."""
    for _ in range(500):
        s = f"{_rng.choice(WHO)} {_rng.choice(DID)} {_rng.choice(WHAT)} {_rng.choice(WHERE)}"
        if s not in used and count_target_pairs(s) >= MIN_TARGET_PAIRS_PER_LOGIN_SENTENCE:
            return s
    return None


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn:
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('enroll', 'login')),
                    features_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS challenges (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    purpose TEXT NOT NULL CHECK (purpose IN ('enroll', 'verify')),
                    sentence TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_samples_user ON samples(username, kind);
                CREATE INDEX IF NOT EXISTS idx_challenges_user ON challenges(username);
                CREATE INDEX IF NOT EXISTS idx_attempts_user ON login_attempts(username, at);
                """
            )


def enroll_count(conn, username: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM samples WHERE username = ? AND kind = 'enroll'", (username,)
    ).fetchone()["c"]


def recent_failures(conn, username: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM login_attempts WHERE username = ? AND success = 0 AND at > ?",
        (username, time.time() - LOCKOUT_WINDOW_SEC),
    ).fetchone()["c"]


# ---------------------------------------------------------------------------
# Input cleaning
# ---------------------------------------------------------------------------
def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sanitize_events(raw) -> list[dict]:
    """Keep only well-formed key events, in time order."""
    if not isinstance(raw, list):
        return []
    clean = []
    for e in raw[:MAX_EVENTS]:
        if not isinstance(e, dict):
            continue
        if e.get("type") not in ("keydown", "keyup") or not _is_number(e.get("t")):
            continue
        clean.append({
            "type": e["type"],
            "key": e.get("key") if isinstance(e.get("key"), str) else "",
            "code": e.get("code") if isinstance(e.get("code"), str) else "",
            "t": float(e["t"]),
            "repeat": e.get("repeat") is True,
        })
    clean.sort(key=lambda e: e["t"])  # stable: keydown stays before keyup on equal times
    return clean


def is_letter(key: str) -> bool:
    return len(key) == 1 and "a" <= key <= "z"


def is_typing_key(key: str) -> bool:
    return is_letter(key) or key == " "


def text_accuracy(typed_text: str, prompt_text: str) -> float:
    if not prompt_text:
        return 0.0
    return difflib.SequenceMatcher(None, typed_text, prompt_text, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def build_presses(events: list[dict]) -> list[dict]:
    """Pair every real keydown with the keyup of the same press."""
    presses, open_by_code = [], {}
    for ev in events:
        code = ev["code"] or ev["key"]
        if ev["type"] == "keydown":
            if ev["repeat"]:  # holding a key fires auto-repeat keydowns; not real presses
                continue
            open_by_code[code] = len(presses)
            presses.append({"key": ev["key"], "down": ev["t"], "up": None})
        else:
            idx = open_by_code.pop(code, None)
            if idx is not None:
                presses[idx]["up"] = ev["t"]
    return presses


def extract_features(events: list[dict], typed_text: str) -> dict:
    """
    digraphs      : letter pair -> list of key-down-to-key-down times (ms), inside words only
    dwell_mean    : average time a letter/space key is held down (ms)
    flight_mean   : average gap from releasing one key to pressing the next (ms, can be negative)
    wpm           : typing speed with thinking pauses removed
    error_rate    : Backspace presses per 100 characters
    rollover_rate : fraction of transitions where the next key is pressed before the last is released
    """
    presses = build_presses(events)

    dwells, flights, intervals = [], [], []
    digraphs: dict[str, list[float]] = {}
    rollovers = 0
    active_ms = 0.0
    char_presses = sum(1 for p in presses if len(p["key"]) == 1)
    backspaces = sum(1 for p in presses if p["key"] == "Backspace")

    for p in presses:
        if is_typing_key(p["key"]) and p["up"] is not None:
            d = p["up"] - p["down"]
            if 0 < d < PAUSE_MS:
                dwells.append(d)

    for prev, nxt in zip(presses, presses[1:]):
        gap = nxt["down"] - prev["down"]
        if gap <= 0 or gap > PAUSE_MS:  # skip thinking pauses
            continue
        active_ms += gap
        intervals.append(gap)
        # A Backspace/Shift between two letters breaks the pair automatically,
        # and a space ends the word, so only clean in-word pairs are kept.
        if is_letter(prev["key"]) and is_letter(nxt["key"]):
            digraphs.setdefault(prev["key"] + nxt["key"], []).append(round(gap, 1))
        if is_typing_key(prev["key"]) and is_typing_key(nxt["key"]) and prev["up"] is not None:
            flight = nxt["down"] - prev["up"]
            if abs(flight) < PAUSE_MS:
                flights.append(flight)
                if flight < 0:
                    rollovers += 1

    wpm = (len(typed_text) / 5.0) / (active_ms / 60000.0) if active_ms > 0 else 0.0

    return {
        "digraphs": digraphs,
        "dwell_mean": round(statistics.fmean(dwells), 2) if dwells else None,
        "flight_mean": round(statistics.fmean(flights), 2) if flights else None,
        "wpm": round(wpm, 2),
        "error_rate": round(100.0 * backspaces / max(char_presses, 1), 2),
        "rollover_rate": round(rollovers / len(flights), 4) if flights else None,
        # used only for quality / bot checks, not stored as identity features
        "_char_presses": char_presses,
        "_dwell_count": len(dwells),
        "_dwell_std": statistics.pstdev(dwells) if len(dwells) > 1 else 0.0,
        "_intervals": intervals,
    }


def quality_problem(features: dict) -> str | None:
    """Reject samples that are incomplete or look machine-generated."""
    if features["dwell_mean"] is None or features["flight_mean"] is None:
        return "Not enough keystroke data. Type the whole sentence."
    if features["_dwell_count"] < 0.7 * features["_char_presses"]:
        return "Key release data missing. Type normally in the box."
    iv = features["_intervals"]
    if len(iv) >= 10:
        mean_iv = statistics.fmean(iv)
        if mean_iv < 35:  # roughly 340+ WPM: not human
            return "Typing too fast to be human."
        if statistics.pstdev(iv) / mean_iv < 0.08:  # real typing is never this even
            return "Typing rhythm is too perfect to be human."
    if features["_dwell_count"] >= 10 and features["_dwell_std"] < 4:
        return "Key hold times are too uniform to be human."
    return None


def storable(features: dict) -> dict:
    return {k: v for k, v in features.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Profile and scoring
# ---------------------------------------------------------------------------
def _mean_std(values: list[float], floor: float) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else floor
    return mean, max(std, floor)


def build_profile(samples: list[dict]) -> dict:
    pooled: dict[str, list[float]] = {}
    for s in samples:
        for pair, times in s["digraphs"].items():
            pooled.setdefault(pair, []).extend(times)
    digraph_profile = {
        pair: _mean_std(times, STD_FLOOR["digraph"])
        for pair, times in pooled.items() if len(times) >= MIN_PAIR_COUNT
    }
    scalar_profile = {}
    for f in SCALAR_FEATURES:
        values = [s[f] for s in samples if s.get(f) is not None]
        if values:
            scalar_profile[f] = _mean_std(values, STD_FLOOR[f])
    return {"digraphs": digraph_profile, "scalars": scalar_profile}


def _z(x: float, mean: float, std: float) -> float:
    return min(abs(x - mean) / std, Z_CAP)


def score_sample(live: dict, profile: dict) -> tuple[float | None, dict]:
    """Weighted average of z-scores. Lower = more like the enrolled user."""
    digraph_z = [
        _z(t, *profile["digraphs"][pair])
        for pair, times in live["digraphs"].items() if pair in profile["digraphs"]
        for t in times
    ]
    details = {"matched_letter_pairs": len(digraph_z)}
    if len(digraph_z) < MIN_OVERLAP:
        return None, details

    parts = {"digraph": statistics.fmean(digraph_z)}
    for f in SCALAR_FEATURES:
        if f in profile["scalars"] and live.get(f) is not None:
            parts[f] = _z(live[f], *profile["scalars"][f])

    total_weight = sum(WEIGHTS[k] for k in parts)
    score = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_weight
    details["feature_z_scores"] = {k: round(v, 3) for k, v in parts.items()}
    return round(score, 4), details


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def parse_submission(purpose: str):
    """Validate the request and consume its one-time challenge. Returns (sample, error)."""
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    token = data.get("token") if isinstance(data.get("token"), str) else ""
    typed_text = data.get("typed_text") if isinstance(data.get("typed_text"), str) else ""
    events = sanitize_events(data.get("events"))

    with closing(get_db()) as conn:
        with conn:
            ch = conn.execute(
                "SELECT * FROM challenges WHERE token = ? AND purpose = ?", (token, purpose)
            ).fetchone()
            if ch is None or ch["used"]:
                return None, "Invalid or already used sentence. Load a new sentence."
            # Single use: burn the token now, whatever the outcome (stops replays).
            conn.execute("UPDATE challenges SET used = 1 WHERE token = ?", (token,))

    if time.time() - ch["issued_at"] > CHALLENGE_TTL_SEC:
        return None, "Sentence expired. Load a new sentence."

    typed_text = typed_text.strip()
    if len(typed_text) < 10:
        return None, "Type the whole sentence."
    if sum(1 for p in build_presses(events) if len(p["key"]) == 1) < MIN_KEYSTROKE_COVERAGE * len(typed_text):
        return None, "Not enough keystroke data. Type the sentence instead of pasting it."
    accuracy = text_accuracy(typed_text, ch["sentence"])
    if accuracy < MIN_ACCURACY:
        return None, f"Typed text matched only {accuracy:.0%} of the sentence. Please retype it."

    features = extract_features(events, typed_text)
    problem = quality_problem(features)
    if problem:
        return None, problem

    return {"username": ch["username"], "sentence": ch["sentence"], "features": features}, None



@app.route("/")
def index():
    return render_template("DNA.html")


@app.route("/prompt")
def get_prompt():
    username = request.args.get("username", "").strip()
    purpose = request.args.get("purpose", "")
    if not username or purpose not in ("enroll", "verify"):
        return jsonify({"error": "username and purpose=enroll|verify are required."}), 400

    with closing(get_db()) as conn:
        with conn:
            done = enroll_count(conn, username)
            progress = None
            if purpose == "enroll":
                if done >= len(ENROLL_SENTENCES):
                    return jsonify({"error": "Already enrolled. Use Log in."}), 409
                sentence = ENROLL_SENTENCES[done]
                progress = f"sentence {done + 1} of {len(ENROLL_SENTENCES)}"
            else:
                if done < len(ENROLL_SENTENCES):
                    return jsonify({"error": f"Finish enrollment first ({done}/{len(ENROLL_SENTENCES)} done)."}), 409
                if recent_failures(conn, username) >= MAX_FAILED_ATTEMPTS:
                    return jsonify({"error": "Too many failed attempts. Try again in 15 minutes."}), 429
                used = {r["sentence"] for r in conn.execute(
                    "SELECT sentence FROM challenges WHERE username = ? AND purpose = 'verify'", (username,))}
                sentence = new_login_sentence(used)
                if sentence is None:
                    return jsonify({"error": "No unused sentences left for this user."}), 503

            token = secrets.token_urlsafe(24)
            conn.execute(
                "INSERT INTO challenges (token, username, purpose, sentence, issued_at) VALUES (?, ?, ?, ?, ?)",
                (token, username, purpose, sentence, time.time()),
            )
    return jsonify({"token": token, "sentence": sentence, "progress": progress})


@app.route("/enroll", methods=["POST"])
def enroll():
    sample, error = parse_submission("enroll")
    if error:
        return jsonify({"status": "error", "reason": error}), 400

    with closing(get_db()) as conn:
        with conn:
            done = enroll_count(conn, sample["username"])
            # Sentences must be typed in order, each exactly once.
            if done >= len(ENROLL_SENTENCES) or sample["sentence"] != ENROLL_SENTENCES[done]:
                return jsonify({"status": "error", "reason": "Out-of-order sentence. Load a new one."}), 409
            conn.execute(
                "INSERT INTO samples (username, kind, features_json, created_at) VALUES (?, 'enroll', ?, ?)",
                (sample["username"], json.dumps(storable(sample["features"])), time.time()),
            )
    done += 1
    return jsonify({
        "status": "enrolled" if done == len(ENROLL_SENTENCES) else "sample saved",
        "username": sample["username"],
        "progress": f"{done}/{len(ENROLL_SENTENCES)}",
    })


@app.route("/verify", methods=["POST"])
def verify():
    sample, error = parse_submission("verify")
    if error:
        return jsonify({"access": "DENIED", "reason": error}), 400
    username = sample["username"]

    with closing(get_db()) as conn:
        if recent_failures(conn, username) >= MAX_FAILED_ATTEMPTS:
            return jsonify({"access": "DENIED", "reason": "Account temporarily locked."}), 429
        rows = conn.execute(
            """SELECT features_json FROM samples WHERE username = ? AND kind = 'enroll'
               UNION ALL
               SELECT features_json FROM (SELECT features_json FROM samples
                   WHERE username = ? AND kind = 'login' ORDER BY id DESC LIMIT ?)""",
            (username, username, MAX_LOGIN_SAMPLES_IN_PROFILE),
        ).fetchall()

    profile = build_profile([json.loads(r["features_json"]) for r in rows])
    score, details = score_sample(sample["features"], profile)

    if score is None:  # not the user's fault: sentence didn't overlap enough with the profile
        return jsonify({"access": "RETRY", "reason": "Not enough matching letter pairs. Load a new sentence.",
                        **(details if SHOW_SCORE_DETAILS else {})})

    if score <= GRANT_THRESHOLD:
        decision = "GRANTED"
    elif score <= STEP_UP_THRESHOLD:
        decision = "STEP_UP"  # borderline: confirm with a second factor such as an OTP
    else:
        decision = "DENIED"

    with closing(get_db()) as conn:
        with conn:
            conn.execute("INSERT INTO login_attempts (username, success, at) VALUES (?, ?, ?)",
                         # only a clear DENIED counts toward the lockout; STEP_UP is gated by the second factor
                         (username, int(decision != "DENIED"), time.time()))
            if decision == "GRANTED":  # adapt slowly as the user's typing changes
                conn.execute(
                    "INSERT INTO samples (username, kind, features_json, created_at) VALUES (?, 'login', ?, ?)",
                    (username, json.dumps(storable(sample["features"])), time.time()),
                )

    result = {"access": decision}
    if SHOW_SCORE_DETAILS:
        result.update({"score": score, "grant_threshold": GRANT_THRESHOLD,
                       "step_up_threshold": STEP_UP_THRESHOLD, **details})
    return jsonify(result)


init_db()  # runs under `python typing_dna.py`, `flask run` and WSGI servers alike

if __name__ == "__main__":
    # debug=True is only safe on 127.0.0.1: the Werkzeug debugger allows code execution if exposed.
    app.run(host="127.0.0.1", port=5000, debug=True)
