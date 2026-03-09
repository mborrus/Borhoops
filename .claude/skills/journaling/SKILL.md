# Journaling

Automatically maintain a daily session log in `docs/` each time work is done on this repo.

## Trigger
Run at the start of every session. Also trigger with /journaling or phrases like "start a journal entry", "log today's work".

## Behavior

1. Check `docs/` for a file matching today's date (`YYYY-MM-DD_*.md`)
2. If none exists, create one:
   - Filename: `docs/YYYY-MM-DD_<short-description>.md`
   - Template:
     ```
     # Session: YYYY-MM-DD — <Short Description>

     ## Carrying over
     <Summarize the "next steps" from the most recent prior session doc>

     ## What was done
     - <filled in as work happens>

     ## Decisions made
     - <filled in as decisions happen>

     ## Next steps
     - <filled in at end of session>
     ```
3. If one already exists for today, read it and continue updating it
4. Read the most recent prior session doc to pull forward context
5. Update the doc throughout the session as work is completed

## Rules
- One file per calendar day
- Keep entries concise — bullet points, not paragraphs
- Don't duplicate info already in `CLAUDE.md` or `golden_rules.md`
- The filename description should reflect the main theme of the session (can be renamed if focus shifts)
