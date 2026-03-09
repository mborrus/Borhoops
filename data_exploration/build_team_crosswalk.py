"""
Build ESPN-to-Kaggle team ID crosswalk.

Three-layer matching:
  1. Manual overrides for ambiguous/abbreviated names (8 teams)
  2. Direct spelling match: lowercase ESPN location → Kaggle TeamSpellings
  3. Unicode normalization fallback: strip accents, retry spelling match

Usage:
  python data_exploration/build_team_crosswalk.py              # build crosswalk
  python data_exploration/build_team_crosswalk.py --validate   # build + validate
"""

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

# ── Paths (relative to repo root) ──────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

ESPN_MENS = REPO_ROOT / "CBBpy/src/cbbpy/utils/mens_team_map.csv"
ESPN_WOMENS = REPO_ROOT / "CBBpy/src/cbbpy/utils/womens_team_map.csv"
KAGGLE_M_TEAMS = REPO_ROOT / "SourceData/static_data/MTeams.csv"
KAGGLE_W_TEAMS = REPO_ROOT / "SourceData/static_data/WTeams.csv"
KAGGLE_M_SPELLINGS = REPO_ROOT / "SourceData/static_data/MTeamSpellings.csv"
KAGGLE_W_SPELLINGS = REPO_ROOT / "SourceData/static_data/WTeamSpellings.csv"
OUTPUT_CSV = REPO_ROOT / "SourceData/derived/espn_kaggle_crosswalk.csv"

# ── Manual overrides: espn_id → kaggle_men_id ──────────────────────────────
# These teams can't be resolved via spelling match due to abbreviation,
# ambiguity, unicode, or naming convention differences.

MANUAL_OVERRIDES = {
    2026:   1111,  # App State → Appalachian St
    2390:   1274,  # Miami → Miami FL (not Miami OH=1193)
    2598:   1384,  # Saint Francis → St Francis PA
    23:     1363,  # San José State → San Jose St (unicode accent)
    2900:   1472,  # St. Thomas-Minnesota → St Thomas MN
    399:    1107,  # UAlbany → SUNY Albany
    2433:   1419,  # UL Monroe → ULM
    292:    1410,  # UT Rio Grande Valley → UTRGV
    2511:   1474,  # Queens University → Queens NC
    2441:   1481,  # New Haven → New Haven (missing from cbbpy team map, ESPN ID confirmed via API)
}

# ESPN IDs to skip — defunct/renamed entries that duplicate an active ID.
# These appear in women's map only and were replaced by a newer ESPN ID.
ESPN_SKIP = {
    2341,  # LIU Brooklyn (women's only, replaced by 112358 Long Island University)
}

# ESPN teams not yet in cbbpy's team map CSVs but confirmed via ESPN API.
# {espn_id: espn_location} — these get injected into the ESPN team set.
ESPN_API_ADDITIONS = {
    2441: "New Haven",  # New D1 team for 2026, ESPN ID confirmed via API
}

MEN_TO_WOMEN_OFFSET = 2000


def strip_accents(text: str) -> str:
    """Remove unicode accents: 'San José' → 'San Jose'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.category(c).startswith("M"))


def load_espn_teams(path: Path) -> dict[int, str]:
    """Return {espn_id: location} for unique ESPN teams (deduplicated across seasons)."""
    teams = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["id"])
            if eid not in teams:
                teams[eid] = row["location"]
    return teams


def load_kaggle_spellings(path: Path) -> dict[str, int]:
    """Return {lowercase_spelling: kaggle_team_id} from TeamSpellings.csv."""
    spellings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            spelling = row["TeamNameSpelling"].strip().lower()
            tid = int(row["TeamID"])
            spellings[spelling] = tid
    return spellings


def load_kaggle_teams(path: Path) -> dict[int, str]:
    """Return {team_id: team_name} from MTeams/WTeams."""
    teams = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams[int(row["TeamID"])] = row["TeamName"]
    return teams


def build_crosswalk() -> list[dict]:
    """Build the ESPN→Kaggle crosswalk using three-layer matching."""
    espn_mens = load_espn_teams(ESPN_MENS)
    espn_womens = load_espn_teams(ESPN_WOMENS)
    m_spellings = load_kaggle_spellings(KAGGLE_M_SPELLINGS)
    w_spellings = load_kaggle_spellings(KAGGLE_W_SPELLINGS)
    m_teams = load_kaggle_teams(KAGGLE_M_TEAMS)
    w_teams = load_kaggle_teams(KAGGLE_W_TEAMS)

    # Merge ESPN men's + women's + API additions for teams missing from cbbpy
    all_espn = {**espn_womens, **espn_mens, **ESPN_API_ADDITIONS}

    rows = []
    for espn_id, location in sorted(all_espn.items()):
        if espn_id in ESPN_SKIP:
            continue
        kaggle_m_id = None
        kaggle_w_id = None
        match_method = "unmatched"

        # Layer 1: Manual override
        if espn_id in MANUAL_OVERRIDES:
            kaggle_m_id = MANUAL_OVERRIDES[espn_id]
            match_method = "override"

        # Layer 2: Direct spelling match
        if kaggle_m_id is None:
            key = location.lower()
            if key in m_spellings:
                kaggle_m_id = m_spellings[key]
                match_method = "spelling"
            elif key in w_spellings:
                kaggle_m_id = w_spellings[key] - MEN_TO_WOMEN_OFFSET
                match_method = "spelling"

        # Layer 3: Unicode normalization fallback
        if kaggle_m_id is None:
            normalized = strip_accents(location.lower())
            if normalized in m_spellings:
                kaggle_m_id = m_spellings[normalized]
                match_method = "normalized"
            elif normalized in w_spellings:
                kaggle_m_id = w_spellings[normalized] - MEN_TO_WOMEN_OFFSET
                match_method = "normalized"

        # Derive women's ID
        if kaggle_m_id is not None:
            kaggle_w_id = kaggle_m_id + MEN_TO_WOMEN_OFFSET

        kaggle_name = m_teams.get(kaggle_m_id, "") if kaggle_m_id else ""

        rows.append({
            "espn_id": espn_id,
            "kaggle_m_id": kaggle_m_id if kaggle_m_id else "",
            "kaggle_w_id": kaggle_w_id if kaggle_w_id else "",
            "espn_location": location,
            "kaggle_team_name": kaggle_name,
            "match_method": match_method,
        })

    return rows


def write_csv(rows: list[dict]) -> None:
    """Write crosswalk to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["espn_id", "kaggle_m_id", "kaggle_w_id", "espn_location",
                  "kaggle_team_name", "match_method"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict]) -> bool:
    """Run validation checks. Returns True if all pass."""
    m_teams = load_kaggle_teams(KAGGLE_M_TEAMS)
    # Filter to currently active Kaggle teams (LastD1Season = 2026)
    active_kaggle_ids = set()
    with open(KAGGLE_M_TEAMS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["LastD1Season"]) >= 2025:
                active_kaggle_ids.add(int(row["TeamID"]))

    matched = [r for r in rows if r["match_method"] != "unmatched"]
    unmatched = [r for r in rows if r["match_method"] == "unmatched"]

    # Method breakdown
    methods = {}
    for r in rows:
        m = r["match_method"]
        methods[m] = methods.get(m, 0) + 1

    print(f"\n{'='*60}")
    print(f"CROSSWALK SUMMARY")
    print(f"{'='*60}")
    print(f"Total ESPN teams:    {len(rows)}")
    print(f"Matched:             {len(matched)}")
    print(f"Unmatched:           {len(unmatched)}")
    print(f"\nBreakdown by method:")
    for method, count in sorted(methods.items()):
        print(f"  {method:15s}  {count}")

    errors = []

    # Check 1: No duplicate ESPN IDs
    espn_ids = [r["espn_id"] for r in rows]
    dupes = [eid for eid in espn_ids if espn_ids.count(eid) > 1]
    if dupes:
        errors.append(f"Duplicate ESPN IDs: {set(dupes)}")

    # Check 2: No duplicate Kaggle men's IDs (among matched)
    kaggle_ids = [r["kaggle_m_id"] for r in matched]
    kaggle_dupes = [kid for kid in kaggle_ids if kaggle_ids.count(kid) > 1]
    if kaggle_dupes:
        errors.append(f"Duplicate Kaggle M IDs: {set(kaggle_dupes)}")

    # Check 3: Women's ID = Men's ID + 2000
    bad_offset = [r for r in matched
                  if r["kaggle_w_id"] != "" and r["kaggle_m_id"] != ""
                  and int(r["kaggle_w_id"]) != int(r["kaggle_m_id"]) + MEN_TO_WOMEN_OFFSET]
    if bad_offset:
        errors.append(f"Bad M→W offset for: {[r['espn_location'] for r in bad_offset]}")

    # Check 4: Every active Kaggle team has an ESPN match
    matched_kaggle = {int(r["kaggle_m_id"]) for r in matched if r["kaggle_m_id"] != ""}
    missing_kaggle = active_kaggle_ids - matched_kaggle
    if missing_kaggle:
        missing_names = [f"{kid} ({m_teams.get(kid, '?')})" for kid in sorted(missing_kaggle)]
        print(f"\nActive Kaggle teams without ESPN match ({len(missing_kaggle)}):")
        for name in missing_names:
            print(f"  {name}")
        # Only error if many are missing — some defunct teams won't have ESPN IDs
        if len(missing_kaggle) > 20:
            errors.append(f"{len(missing_kaggle)} active Kaggle teams have no ESPN match")

    if unmatched:
        print(f"\nUnmatched ESPN teams:")
        for r in unmatched:
            print(f"  ESPN {r['espn_id']:>5}  {r['espn_location']}")

    if errors:
        print(f"\n{'!'*60}")
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print(f"{'!'*60}")
        return False

    print(f"\nAll validation checks passed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build ESPN-to-Kaggle team ID crosswalk")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    args = parser.parse_args()

    # Verify source files exist
    for path in [ESPN_MENS, ESPN_WOMENS, KAGGLE_M_TEAMS, KAGGLE_W_TEAMS,
                 KAGGLE_M_SPELLINGS, KAGGLE_W_SPELLINGS]:
        if not path.exists():
            print(f"ERROR: Missing source file: {path}", file=sys.stderr)
            sys.exit(1)

    rows = build_crosswalk()
    write_csv(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    if args.validate:
        ok = validate(rows)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
