@AGENTS.md

## Claude Code only
- Durable project knowledge (rules, decisions, current state) goes in AGENTS.md or docs/, not auto-memory, so every agent sees it. Keep auto-memory for notes that only matter to Claude Code.
- The auto-mode classifier blocks `kubectl apply/delete` and similar cluster changes. Don't work around it: hand the commands over as AGENTS.md *Ask first* says.
- `.claude/` is gitignored here, so `.claude/settings.local.json` (personal allow rules) is the only settings file; nothing in it is shared.
