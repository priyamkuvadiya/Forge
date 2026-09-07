"""Facts about the sandbox that hold on every platform.

Separate from `test_code_exec.py`, which skips wholesale when no sandbox
exists. These tests run anywhere, and that is the point: the thing most worth
asserting about the POSIX backend from a Windows machine is what it *claims*
to confine, because the claim is what the eval harness will record next to a
coding score.
"""

import sys

import pytest

from tools import code_exec, code_exec_posix
from tools.code_exec import (
    SandboxError,
    confinement,
    main,
    sandbox_available,
    sandbox_backend,
)


def test_the_backend_matches_the_platform():
    backend = sandbox_backend()
    if sys.platform == "win32":
        assert backend == "windows"
    elif __import__("os").name == "posix":
        assert backend == "posix"
    else:
        assert backend is None
    assert sandbox_available() == (backend is not None)


def test_the_posix_backend_imports_anywhere():
    """`resource` is POSIX-only and is imported lazily so this holds.

    Without it, a Windows machine could not read the POSIX backend's declared
    confinement, lint it, or have CI see the file at all - and this project is
    developed entirely on Windows.
    """
    assert code_exec_posix.CONFINEMENT.level == "rlimit-only"


def test_the_posix_backend_admits_it_does_not_confine_the_filesystem():
    """The honest entry, asserted so it cannot be quietly upgraded.

    If someone later adds Landlock or a mount namespace, this test fails and
    forces the claim to be updated deliberately rather than by accident. Until
    then a submission on POSIX can read the repo, and therefore the expected
    answers module 2 withholds from the harness.
    """
    posix = code_exec_posix.CONFINEMENT
    assert posix.filesystem is False
    assert posix.network is False
    assert posix.resources is True
    assert "expected answers" in posix.note


@pytest.mark.skipif(sys.platform != "win32", reason="Windows backend")
def test_the_windows_backend_claims_full_confinement():
    detail = confinement()
    assert detail.filesystem and detail.network and detail.resources


def test_confinement_is_serialisable():
    """Module 9 records this beside every coding score, so it has to be data."""
    import json

    if not sandbox_available():
        pytest.skip("no backend to report on")
    payload = json.loads(json.dumps(confinement().to_dict()))
    assert set(payload) == {"level", "filesystem", "network", "resources", "note"}


def test_summary_names_what_is_confined():
    assert "resources" in code_exec_posix.CONFINEMENT.summary()
    assert "filesystem" not in code_exec_posix.CONFINEMENT.summary()


def test_an_unsupported_platform_raises_rather_than_failing_every_call(monkeypatch):
    """A missing sandbox must be loud.

    `verify()` refuses to score a coding task without a runner for the same
    reason: a sandbox that silently failed every call would flatten the code
    category to zero and look exactly like a model that cannot code.
    """
    monkeypatch.setattr(code_exec, "sandbox_backend", lambda: None)
    with pytest.raises(SandboxError):
        code_exec.run_code("print(1)")


def test_the_cli_reports_the_backend_without_doing_setup(capsys):
    """`--setup` is opt-in; bare invocation must stay fast."""
    assert main([]) == 0

    printed = capsys.readouterr().out
    assert "backend:" in printed
    assert "confinement:" in printed
    assert "setup:" not in printed


@pytest.mark.skipif(sys.platform == "win32", reason="Windows does confine the filesystem")
def test_the_cli_warns_loudly_when_the_filesystem_is_not_confined(capsys):
    main([])
    assert "WARNING" in capsys.readouterr().out
