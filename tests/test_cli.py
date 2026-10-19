"""CLI command registration."""

import pytest
from click.testing import CliRunner

from agentbench.cli.main import cli

EXPECTED_COMMANDS = (
    "run",
    "compare",
    "trace",
    "failures",
    "report",
    "regression",
    "optimize",
)


def test_all_commands_are_registered() -> None:
    assert set(cli.commands) == set(EXPECTED_COMMANDS)


@pytest.mark.parametrize("name", EXPECTED_COMMANDS)
def test_command_help(name: str) -> None:
    result = CliRunner().invoke(cli, [name, "--help"])
    assert result.exit_code == 0, result.output
    assert name in result.output
