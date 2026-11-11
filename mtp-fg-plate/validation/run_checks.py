"""Run the complete delivered verification and save readable logs."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT/'evidence'
    evidence.mkdir(exist_ok=True)
    jobs = [
        ('mechanics', ['validation/validate.py', '--full']),
        ('extensions_and_pilot', ['-m', 'unittest', 'discover', '-s', 'validation', '-p', 'test_*.py', '-v']),
        ('saved_artifact_audit', ['validation/check_artifacts.py']),
    ]
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), commands=[])
    for name, arguments in jobs:
        completed = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT, capture_output=True, text=True)
        text = completed.stdout+'\n--- STDERR (warnings preserved) ---\n'+completed.stderr
        (evidence/(name+'.txt')).write_text(text, encoding='utf-8')
        result['commands'].append(dict(name=name, arguments=arguments, exit_code=completed.returncode))
        print(f'{name}: {"PASS" if completed.returncode == 0 else "FAIL"}', flush=True)
        if completed.returncode: print(text, flush=True)
    (evidence/'verification_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return 0 if all(row['exit_code'] == 0 for row in result['commands']) else 1


if __name__ == '__main__': sys.exit(main())
