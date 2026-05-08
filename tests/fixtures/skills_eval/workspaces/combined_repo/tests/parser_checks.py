from src.parser import parse_text


def parser_case_strips_outer_whitespace() -> None:
    assert parse_text("  ok  ") == "ok"
