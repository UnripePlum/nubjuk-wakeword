import io
from contextlib import redirect_stdout

from nubjuk_wakeword import cli


def test_check_env_command_returns_zero_and_prints_stub() -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["check-env"])

    output = stream.getvalue()
    assert code == 0
    assert "[check-env]" in output


def test_synth_command_returns_zero_and_prints_stub() -> None:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = cli.main(["synth"])

    output = stream.getvalue()
    assert code == 0
    assert "[synth]" in output
