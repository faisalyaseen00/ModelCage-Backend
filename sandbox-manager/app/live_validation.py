"""Run the real integration suite and retain a strict, auditable result."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def summarize(report, returncode):
    """A skipped, empty, interrupted or failed run never validates capture."""
    counts = dict(tests=0, passed=0, failures=0, errors=0, skipped=0)
    details = []
    try:
        root = ET.parse(report).getroot()
        for case in root.iter('testcase'):
            counts['tests'] += 1
            outcome = 'passed'
            for tag, key in [('failure', 'failures'), ('error', 'errors'), ('skipped', 'skipped')]:
                element = case.find(tag)
                if element is not None:
                    outcome = key
                    details.append(dict(test=case.get('name'), outcome=key,
                                        reason=element.get('message', '')[:2000]))
                    break
            counts[outcome] += 1
    except (OSError, ET.ParseError) as error:
        details.append(dict(outcome='report_error', reason=str(error)))
    validated = (returncode == 0 and counts['tests'] > 0 and
                 counts['passed'] == counts['tests'] and not details)
    return dict(validated=validated, pytest_exit_code=returncode, **counts, details=details)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New report directory (must not already exist)')
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    if not (project / 'tests' / 'test_live_capture_coverage.py').is_file():
        parser.error('Run from an editable ModelCage source installation containing tests')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'pytest', str(project / 'tests'), '-m', 'integration',
               '-q', '-rs', '-p', 'no:cacheprovider', '--junitxml', str(output / 'junit.xml'),
               '--basetemp', str(output / 'artifacts')]
    started = datetime.now(timezone.utc).isoformat()
    # Inherit the caller's authorized runsc wrapper/root and Docker environment.
    # No sudo, permission changes, fallback tracer, or skipped-test exemptions.
    with (output / 'pytest.log').open('w') as log:
        process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    result = summarize(output / 'junit.xml', returncode)
    result.update(started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
                  scope='Controlled pickle-loading workloads, exact syscall counts, process/thread '
                        'attribution, concurrency, failure paths, cleanup and CSV routing. '
                        'Not exhaustive syscall or representative ML inference coverage.',
                  command=command)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Validation {'PASSED' if result['validated'] else 'INCOMPLETE/FAILED'}: {output / 'summary.json'}")
    return 0 if result['validated'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
