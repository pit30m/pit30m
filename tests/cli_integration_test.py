
from pit30m.cli import Pit30MCLI


def test_cli_knows_log_ids():
    cli = Pit30MCLI()
    assert len(cli.all_log_ids) == 1518
