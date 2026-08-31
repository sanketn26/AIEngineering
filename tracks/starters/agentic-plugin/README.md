# Starter — agentic editor plugin (Python slice)

**Not the 90-day VS Code extension.** One command (`explain_selection`) goes through a **mock model** that may only propose `read_file`. The runtime allowlists the tool and **never writes**.

Full track: [docs/tracks/agentic-plugin.md](../../../docs/tracks/agentic-plugin.md).

```bash
cd tracks/starters/agentic-plugin
python3 -c "from plugin import handle_command; print(handle_command('explain_selection', {'path': 'fixtures/hello.txt'}))"
python3 -m pytest tests/test_slice.py -v
```

**Model proposes. Runtime disposes.** If that sentence is not true in code, you do not have an agent — you have `eval`.
