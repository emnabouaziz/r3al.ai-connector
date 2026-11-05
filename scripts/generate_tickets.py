"""Generate the synthetic support-ticket export (CSV + re-exported .eml).

    python scripts/generate_tickets.py [--config config.production.yaml]

Paths (csv_path, eml_dir), seed and PII manifest come from the config.
Reproducible: same seed -> identical bytes.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from src.config import load_config
from src.synthetic.tickets import generate_tickets

logger = logging.getLogger("generate_tickets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.production.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")

    config = load_config(args.config)
    email_sources = [s for s in config.get("sources", []) if s.get("type") == "email"]
    if not email_sources:
        logger.error("generate_failed reason=no_email_source_in_config")
        return 1
    source_config = email_sources[0]["config"]
    csv_path, eml_dir = Path(source_config["csv_path"]), Path(source_config["eml_dir"])
    if eml_dir.parent != csv_path.parent or eml_dir.name != "emails":
        logger.error("generate_failed reason=eml_dir_must_be_<csv folder>/emails")
        return 1
    manifest = (config.get("verify") or {}).get("pii_manifest")

    written_csv, emls = generate_tickets(csv_path.parent, int(config["seed"]), manifest)
    digest = hashlib.sha256(written_csv.read_bytes()).hexdigest()[:12]
    logger.info("tickets_written csv=%s bytes=%d sha256=%s", written_csv.name, written_csv.stat().st_size, digest)
    eml_digest = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(emls))).hexdigest()[:12]
    logger.info("eml_written files=%d sha256_all=%s manifest_updated=%s", len(emls), eml_digest, bool(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
