# CBBpy Code Review

Full review of the CBBpy repository (`/Users/marshallborrus/Desktop/OfficialRepos/Borhoops/CBBpy`), conducted 2026-02-27.

---

## Bugs (things that will crash at runtime)

1. **`_get_player_info` returns `None` instead of a DataFrame** — When a player page 404s, the function `break`s out of the retry loop but `df` is initialized to `None`, so callers get `None` instead of an empty DataFrame (`cbbpy_utils.py:600-657`)

2. **`_get_player_info` accesses `soup.text` without a None guard** — If the HTTP request itself fails (before BeautifulSoup parses), `soup` is still `None` and `soup.text` throws `AttributeError`. Every other scraping function checks `if soup is not None:` first (`cbbpy_utils.py:622-626`)

3. **`_get_games_range` returns `()` instead of a 3-tuple when there's no data** — Any user doing `info, box, pbp = ms.get_games_season(2025)` gets `ValueError: not enough values to unpack` (`cbbpy_utils.py:212`)

4. **PBP parser crashes when plays have no clock data** — `time_splits` gets `""` (empty string) for clockless plays, then `int(""[0])` raises `IndexError` (`cbbpy_utils.py:1071-1078`)

5. **`os.cpu_count()` can return `None`** — On some platforms (containers, certain VMs), `None - 1` raises `TypeError` (`cbbpy_utils.py:180, 264`)

6. **`.iloc[0]` on possibly empty DataFrame in `_get_games_team`** — If `_get_team_schedule` returns an empty DataFrame, `schedule_df.team.iloc[0]` raises `IndexError` (`cbbpy_utils.py:268`)

7. **`pd.concat` on empty list in `_get_games_team` and `_get_games_conference`** — A team with no completed games produces an empty result list, and `pd.concat([])` raises `ValueError` (`cbbpy_utils.py:276, 284, 292`)

8. **`re.search().group(1)` without None check** — If ESPN changes their page format and the regex doesn't match, `None.group(1)` throws `AttributeError` at 4 different call sites (`cbbpy_utils.py:1603, 1617, 1632, 1647`)

---

## Design Issues

9. **`pnf_` is a module-level mutable list that's never cleared** — Game IDs that 404'd in one session are silently skipped in all subsequent calls. It's also written to from joblib parallel workers, which is a race condition waiting to happen (`cbbpy_utils.py:138`)

10. **`_call_depth` counter isn't thread-safe** — Used under `joblib.Parallel`, so concurrent workers can corrupt the counter (`cbbpy_utils.py:120-133`)

11. **Team map CSV is re-read from disk on nearly every call** — `_get_team_map()` reads the CSV each time. With conference queries calling it dozens of times, this is a lot of unnecessary I/O. Should be cached with `functools.lru_cache` or similar (`cbbpy_utils.py:1488-1544`)

12. **`cbbpy_utils.py` is a ~1700-line monolith** — All scraping, parsing, caching, logging, date handling, fuzzy matching, and data transformation lives in one file. Breaking it into modules (e.g., `http.py`, `parsers.py`, `matching.py`) would make it much easier to maintain

13. **~110 lines of copy-pasted code for team1 vs team2 boxscore processing** — The entire block for building team1's boxscore DataFrame is duplicated verbatim for team2. A helper function would eliminate this (`cbbpy_utils.py:748-994`)

14. **`mens_scraper.py` and `womens_scraper.py` are identical except for a string** — Every function in both files is a thin wrapper that passes `"mens"` or `"womens"` to `cbbpy_utils`. This could be a single parameterized class or a factory function, cutting 273 lines of duplication

---

## Error Handling

15. **Bare `except:` in `_parse_date`** — Catches `KeyboardInterrupt`, `SystemExit`, etc. Should be `except ValueError:` (`cbbpy_utils.py:733`)

16. **Bare `except:` in odds parsing** — Same issue. Should be `except (KeyError, IndexError, TypeError):` (`cbbpy_utils.py:1318-1321`)

17. **Wrong error labels in `pnf_` checks** — When `info=False`, the code still logs `"Game Info: Page not found error"` even though game info was never requested (`cbbpy_utils.py:153-168`)

---

## Testing Gaps

18. **4 of 13 public functions have zero test coverage** — `get_game()`, `get_games_team()`, `get_game_ids()`, and `get_teams_from_conference()` are not tested at all

19. **No edge case tests** — Invalid game IDs, empty results, malformed ESPN pages, and fuzzy matching edge cases are untested

20. **No mock/unit tests** — All tests hit real ESPN endpoints (integration tests). A network outage or ESPN format change breaks the entire test suite. Mock tests for the parsing logic would be more reliable

21. **`get_games_season()` only tests the error path** — The happy path (actually scraping a season) is never tested

---

## Code Quality

22. **Commented-out dead code** — Shot chart columns (`shotteams`, `shotdescs`) and a debug print are commented out but left in production code (`cbbpy_utils.py:1194-1202, 300`)

23. **No type hints anywhere** — The entire codebase has zero type annotations. Adding them would catch bugs at dev time and improve IDE support

24. **`egg-info` directory is checked into git** — `src/CBBpy.egg-info/` is a build artifact that should be in `.gitignore`

25. **CI installs flake8 and mypy but never runs them** — The pytest workflow installs linting and type-checking tools but only runs `pytest tests/`. Those tools aren't doing anything

---

## Packaging & CI

26. **GitHub Actions uses outdated action versions** — `actions/checkout@v3`, `actions/setup-python@v3`/`v4`, and `actions/cache@v3` are all behind current versions (v4+). The older versions may stop working as GitHub deprecates Node.js 16 runners

27. **PyPI publish action is pinned to a specific commit hash** — `pypa/gh-action-pypi-publish@27b31...` is good for security, but it's an old version that should be periodically updated

28. **No dev dependencies specified** — `pytest`, `flake8`, `mypy` are installed ad-hoc in CI. A `[project.optional-dependencies]` section in `pyproject.toml` (e.g., `dev = ["pytest", "flake8", "mypy"]`) would standardize this

29. **Tests only run on PRs to master** — Tests don't run on pushes to feature branches, so you only find out about failures when you open the PR

---

## Priority Summary

- **Items 1-8** (Bugs): Highest priority — runtime crashes users will hit during normal usage
- **Items 9-14** (Design): Structural improvements that would make the codebase significantly easier to maintain
- **Items 15-17** (Error Handling): Quick wins that improve robustness
- **Items 18-21** (Testing): Important for long-term reliability
- **Items 22-29** (Quality/CI): Cleanup and modernization
