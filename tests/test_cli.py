from agent_tools.cli import main


def test_cli_reports_connection_error(capsys) -> None:
    status = main(["--host", "http://127.0.0.1:1", "--prompt", "Hello"])
    assert status == 1
    assert "Agent error:" in capsys.readouterr().err


def test_cli_rejects_empty_prompt(capsys) -> None:
    assert main(["--prompt", ""]) == 1
    assert "must not be empty" in capsys.readouterr().err
