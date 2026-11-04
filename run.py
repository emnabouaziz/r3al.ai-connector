"""Command-line entry point.

    python run.py --config config.yaml

Exit codes: 0 success, 2 leakage detected, 1 any other run failure.
"""

import argparse
import logging
import sys

from src.config import ConfigError, load_config
from src.counters import InvariantViolation
from src.dataset.leakage import LeakageError
from src.processing.pipeline import SourceUnavailable, run_pipeline
from src.registry import RegistryError

logger = logging.getLogger("run")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_LEAKAGE = 2


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build train/eval datasets.")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--debug", action="store_true", help="re-raise unexpected errors with a traceback"
    )
    args = parser.parse_args(argv)
    configure_logging(args.log_level.upper())

    try:
        result = run_pipeline(load_config(args.config))
    except LeakageError as exc:
        # Our own message: ids and counts only.
        logger.error("run_failed reason=leakage detail=%s", exc)
        return EXIT_LEAKAGE
    except (ConfigError, RegistryError, SourceUnavailable, InvariantViolation) as exc:
        logger.error("run_failed reason=%s detail=%s", type(exc).__name__, exc)
        return EXIT_FAILURE
    except Exception as exc:
        # Unknown messages may contain content: log the class only unless --debug.
        if args.debug:
            raise
        logger.error("run_failed reason=unexpected error=%s", type(exc).__name__)
        return EXIT_FAILURE

    logger.info(
        "run_ok train=%d eval=%d unmeasurable_domains=%s",
        len(result.train),
        len(result.eval),
        ",".join(result.split.unmeasurable_domains) or "-",
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
