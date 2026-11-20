from pathlib import Path

from agent_tools.tools import calculate, get_current_time, make_file_tools


def test_calculate_basic_expressions() -> None:
    assert calculate.run({"expression": "(2 + 3) * 4"}) == "20"
    assert calculate.run({"expression": "-5 + 2.5"}) == "-2.5"
    assert calculate.run({"expression": "2 ** 8"}) == "256"


def test_calculate_rejects_unsafe_expressions() -> None:
    out = calculate.run({"expression": "__import__('os').system('echo hi')"})
    assert out.startswith("Error executing 'calculate':")


def test_get_current_time() -> None:
    utc_time = get_current_time.run({"timezone": "UTC"})
    assert "T" in utc_time
    unknown_tz = get_current_time.run({"timezone": "Mars/Olympus"})
    assert "Unknown timezone" in unknown_tz
    assert unknown_tz.startswith("Error executing")


def test_file_tools_scoped_to_workspace(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested", encoding="utf-8")

    list_files, read_file = make_file_tools(tmp_path)

    assert list_files.run({"directory": "."}) == '["hello.txt", "sub/"]'
    assert read_file.run({"path": "hello.txt"}) == "hello world"
    assert read_file.run({"path": "sub/nested.txt"}) == "nested"

    traversal = read_file.run({"path": "../etc/passwd"})
    assert "Access denied" in traversal


def test_file_tools_reject_symlink_escape_and_truncate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(secret)
    (workspace / "large.txt").write_text("hello world", encoding="utf-8")
    _, read_file = make_file_tools(workspace)

    assert "Access denied" in read_file.run({"path": "link.txt"})
    assert read_file.run({"path": "large.txt", "max_chars": 5}) == "hello\n... [truncated]"
    assert read_file.run({"path": "large.txt", "max_chars": 11}) == "hello world"


def test_calculate_limits_large_results_and_complex_numbers() -> None:
    assert "too large" in calculate.run({"expression": "(2 ** 1000) ** 1000"})
    assert "Complex numbers" in calculate.run({"expression": "(-1) ** 0.5"})
