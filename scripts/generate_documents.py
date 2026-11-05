"""Generate the synthetic, deliberately messy document folder.

    python scripts/generate_documents.py [--config config.production.yaml]

The seed, the target folder (the "files" source root) and the PII manifest
path all come from the config. Reproducible: same seed -> identical bytes.
"""

import argparse
import hashlib
import logging
import sys

from src.config import load_config
from src.synthetic.documents import generate_documents

logger = logging.getLogger("generate_documents")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.production.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")

    config = load_config(args.config)
    files_sources = [s for s in config.get("sources", []) if s.get("type") == "files"]
    if not files_sources:
        logger.error("generate_failed reason=no_files_source_in_config")
        return 1
    root = files_sources[0]["config"]["root"]
    manifest = (config.get("verify") or {}).get("pii_manifest")

    written = generate_documents(root, int(config["seed"]), manifest)
    for path in written:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        logger.info("document_written file=%s bytes=%d sha256=%s", path.name, path.stat().st_size, digest)
    logger.info("generate_ok files=%d root=%s manifest_updated=%s", len(written), root, bool(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
