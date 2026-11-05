"""Command-line entry point.

    python run.py --config config.yaml

The pipeline writes train.jsonl / eval.jsonl / quality report into
out/.staging/. Two blocking PII scans then run on the staged files:
  1. pattern scan: the anonymiser's validated patterns, re-run on the output;
  2. ground-truth scan: literal search for every PII value the synthetic
     generator injected (only when its manifest exists).
Files are promoted to out/ only if both pass. On failure they stay in
out/.staging/ for inspection: a false positive never destroys a run.

Exit codes:
  0 success
  1 any other run failure
  2 train/eval leakage detected
  3 PII found by the pattern scan
  4 PII found by the ground-truth scan (anonymisation recall < 100%)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_config
from src.counters import InvariantViolation
from src.dataset.leakage import LeakageError
from src.processing.pipeline import PipelineResult, SourceUnavailable, promote_outputs, run_pipeline
from src.registry import RegistryError
from src.report.ground_truth_scan import RecallResult, no_ground_truth_report, scan_against_manifest
from src.report.pii_scan import scan_output_files
from src.report.quality_report import write_quality_report

logger = logging.getLogger("run")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_LEAKAGE = 2
EXIT_PII = 3
EXIT_PII_GROUND_TRUTH = 4
MAX_FINDINGS_LOGGED = 50


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _ground_truth(config: dict[str, Any], result: PipelineResult, staged: list[str]) -> RecallResult | None:
    verify = config.get("verify") or {}
    manifest = verify.get("pii_manifest") if isinstance(verify, dict) else None
    if not manifest or not Path(str(manifest)).is_file():
        logger.info("stage=ground_truth_scan skipped reason=no_manifest")
        return None
    recall = scan_against_manifest(staged, str(manifest), result.ingested_sources)
    logger.info(
        "stage=ground_truth_scan injected=%d leaked=%d overall_recall=%.4f per_type=%s",
        sum(recall.total_injected.values()), recall.leaked, recall.overall_recall,
        ",".join(f"{t}:{r:.4f}" for t, r in sorted(recall.recall.items())),
    )
    return recall


def publish(config: dict[str, Any], result: PipelineResult) -> int:
    """Run both PII scans on the staged outputs; promote them only if both pass."""
    staged = [str(result.outputs["train"]), str(result.outputs["eval"])]
    findings = scan_output_files(staged)
    recall = _ground_truth(config, result, staged)

    result.report["anonymisation_recall"] = (
        recall.as_report() if recall is not None
        else no_ground_truth_report("No PII manifest available (e.g. real client data): recall not measurable.")
    )
    write_quality_report(result.report, result.staging_dir)

    if findings:
        for finding in findings[:MAX_FINDINGS_LOGGED]:
            logger.error("pii_found scan=pattern file=%s line=%d type=%s",
                         Path(finding.file).name, finding.line, finding.pii_type)
        logger.error("run_failed reason=pii_in_output scan=pattern findings=%d staged_outputs=%s",
                     len(findings), result.staging_dir)
        return EXIT_PII
    logger.info("stage=pii_scan files=%d findings=0", len(staged))

    if recall is not None and recall.leaked:
        for miss in recall.misses[:MAX_FINDINGS_LOGGED]:
            logger.error("pii_found scan=ground_truth file=%s line=%d type=%s value_hash=%s",
                         miss["file"], miss["line"], miss["type"], miss["value_hash"])
        logger.error("run_failed reason=pii_in_output scan=ground_truth leaked_values=%d staged_outputs=%s",
                     recall.leaked, result.staging_dir)
        return EXIT_PII_GROUND_TRUTH

    promote_outputs(result)
    logger.info("stage=publish out_dir=%s files=%d", result.out_dir, len(result.outputs))
    return EXIT_OK


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
        config = load_config(args.config)
        result = run_pipeline(config)
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

    code = publish(config, result)
    if code == EXIT_OK:
        logger.info(
            "run_ok train=%d eval=%d not_measurable=%s",
            len(result.train),
            len(result.eval),
            ",".join(f"{d}:{s}" for d, s in sorted(result.split.domain_status.items())
                     if s != "MEASURABLE") or "-",
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
