# config.py
# ─────────────────────────────────────────────────────────────────────────────
# NPS Fraud Detection Engine — All Thresholds & Column Mappings
# Trends Retail / Reliance Retail — PRD v1.0
# ─────────────────────────────────────────────────────────────────────────────

# ── Column names in your CSV ──────────────────────────────────────────────────
RRID_COL        = "response_id"       # Phone number / respondent unique ID
STORE_COL       = "entity_id"         # Store code
DATE_COL        = "response_date"     # Survey date
NPS_COL         = "nps_score"         # NPS score (0–10)
FEEDBACK_COL    = "other_feedback"    # Top-level feedback column (may be empty)
ANSWERS_COL     = "answers"           # JSON column containing all sub-ratings + verbatim

# ── Sub-rating question IDs inside the answers JSON ───────────────────────────
# These are the numeric 1–5 rating questions
SUB_RATING_QUESTIONS = [
    "staff_friendliness_service",
    "staff_rating",
    "billing_experience",
    "billing_rating",
    "product_size_availability",
    "product_rating",
    "store_ambience",
    "ambience_rating",
    "trial_room_experience",
    "product_options_variety",
]

# The verbatim (text) question ID inside the answers JSON
VERBATIM_QUESTION_ID = "other_feedback"

# ── Layer 1: Duplicate RRID ───────────────────────────────────────────────────
L1_WINDOW_DAYS         = 7    # Rolling window days for duplicate check
L1_HEAVY_DUP_THRESHOLD = 3    # 3+ submissions from same RRID → ALL flagged
L1_LIGHT_DUP_THRESHOLD = 2    # Exactly 2 → only 2nd flagged

# ── Layer 2: Store Contamination ─────────────────────────────────────────────
L2_WINDOW_DAYS               = 7      # Rolling window days
L2_DUP_RATIO_THRESHOLD       = 0.30   # >30% responses from repeat RRIDs in window
L2_HEAVY_DUP_COUNT_MIN       = 5      # ≥5 responses from RRIDs submitting 3+ times in window
L2_ALL_PERFECT_RATE          = 0.90   # >90% all-perfect rate in window
L2_ALL_PERFECT_MIN_RESPONSES = 15     # Minimum responses in window before rate check applies
ALL_PERFECT_MIN_RATING       = 5      # Sub-rating >= this = counts toward all-perfect
ALL_PERFECT_MIN_NPS          = 10     # NPS >= this = counts toward all-perfect

# ── Layer 3: Response Velocity Anomaly ───────────────────────────────────────
L3_MIN_RESPONSES         = 6     # ≥6 responses in a store-day (P95 of clean stores)
L3_UNIQUE_RRID_RATIO_MAX = 0.70  # <70% unique RRIDs (few people, many responses)
L3_PERFECT_RATE_MIN      = 0.80  # >80% all-perfect on that store-day

# ── Layer 4: Feedback Text Fingerprinting ────────────────────────────────────
L4_RRID_EXACT_THRESHOLD   = 2   # Same RRID, exact same text ≥2 times → fraud
L4_RRID_SIMILAR_THRESHOLD = 2   # Same RRID, similar text ≥2 times → fraud
L4_STORE_EXACT_THRESHOLD  = 3   # Same store, exact same text ≥3 times → fraud
L4_STORE_SIMILAR_THRESHOLD = 4  # Same store, similar text ≥4 times → fraud
# Lazy/genuine single-word phrases to EXCLUDE from copy-paste detection
L4_EXCLUDED_PHRASES = {
    "good", "nice", "ok", "okay", "excellent", "very good", "super",
    "no", "all good", "good service", "thank you", "nothing", "great",
    "best", "fine", "perfect", "awesome", "good experience", "nice store",
    "good store", "very nice", "very excellent", "superb", "outstanding",
    "fantastic", "wonderful", "amazing", "satisfied", "happy", "", "na",
    "n/a", "nil", "none", "bahut acha", "bahut achha", "accha", "achha",
    # Extended generic phrases
    "very good experience", "nice experience", "too good", "very helpful",
    "good work", "keep it up", "well done", "good job",
    "nice experience overall", "overall good experience",
    "satisfied with service", "good service overall",
    # Hindi/transliterated generic phrases
    "bht acha", "bohot acha", "bohot achha", "acha",
    "sab sahi hai", "sab theek hai", "theek hai", "bilkul sahi",
    # More English variants
    "overall good", "all good overall", "good overall",
    "everything good", "everything was good", "everything is good",
    "nice shop", "good shop", "good store experience", "nice store experience",
    "happy with service", "happy with experience",
    "completely satisfied", "fully satisfied",
    "no complaints", "no issues", "no problem", "no problems",
    "carry on", "keep up the good work",
}

# ── Layer 5: Scoring Contradictions ──────────────────────────────────────────
L5_MONOTONE_LOW_MAX          = 3.0   # Sub-ratings all ≤ this, but NPS = 10
L5_EXTREME_CONTRADICTION_AVG = 3.0   # Avg sub-rating ≤ this AND NPS ≥ 9
L5_EXTREME_NPS_MIN           = 9
L5_REVERSE_AVG_MIN           = 4.5   # Avg sub-rating ≥ this AND NPS ≤ 3
L5_REVERSE_NPS_MAX           = 3

# ── Staff Coaching Detection ─────────────────────────────────────────────────
# Same staff name mentioned 3+ times at same store = coaching signal
COACHING_NAME_MIN_COUNT = 3

# ── Composite Fraud Scoring ───────────────────────────────────────────────────
# Weights per layer (L1 and L5 are definitive, higher weight)
LAYER_WEIGHTS = {
    "layer1": 35,
    "layer2": 20,
    "layer3": 20,
    "layer4_rrid_exact":   40,
    "layer4_rrid_similar": 35,
    "layer4_store_exact":  25,
    "layer4_store_similar": 15,
    "layer5": 35,
}
IS_FRAUD_THRESHOLD = 1   # Flag if ≥1 layer triggered

# ── BigQuery ─────────────────────────────────────────────────────────────────
BQ_PROJECT_ID = "fynd-jio-impetus-prod"
BQ_DATASET    = "nps_data"
BQ_TABLE      = "nps_responses_06_03_2026"
BQ_QUERY      = (
    f"SELECT {RRID_COL}, {STORE_COL}, {DATE_COL}, {NPS_COL}, {ANSWERS_COL}, {FEEDBACK_COL}, "
    f"entity_geo "
    f"FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` "
    f"WHERE account_id = 1"
)

# ── NPS Ranges ────────────────────────────────────────────────────────────────
NPS_PROMOTER_MIN  = 9
NPS_PASSIVE_MIN   = 7
NPS_DETRACTOR_MAX = 6

