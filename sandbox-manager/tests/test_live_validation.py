import pytest

from app.live_validation import summarize


@pytest.mark.parametrize('body,code,expected', [
    ('<testcase name="ok"/>', 0, True),
    ('<testcase><skipped message="no access"/></testcase>', 0, False),
    ('<testcase><failure message="wrong count"/></testcase>', 1, False),
    ('<testcase><error message="setup"/></testcase>', 1, False),
    ('<testcase/>', 2, False),
    ('', 0, False),
])
def test_strict_validation_report(tmp_path, body, code, expected):
    report = tmp_path / 'junit.xml'
    report.write_text(f'<testsuites><testsuite>{body}</testsuite></testsuites>')
    assert summarize(report, code)['validated'] is expected


def test_missing_or_truncated_report_is_incomplete(tmp_path):
    report = tmp_path / 'junit.xml'
    assert not summarize(report, 0)['validated']
    report.write_text('<testsuites>')
    assert not summarize(report, 0)['validated']
