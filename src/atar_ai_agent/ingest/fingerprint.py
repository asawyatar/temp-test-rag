"""The value that tells ingestion whether a source needs re-processing."""

import hashlib


def fingerprint(data: bytes, *, chunker_version: str, embedding_model: str) -> str:
    """Hash a source's bytes together with how it will be processed.

    Two runs give the same fingerprint only if the file is byte-identical AND
    it would be chunked and embedded the same way. Changing the chunker or the
    embedding model therefore re-processes every source on the next run,
    without anyone having to remember to force it.
    """
    digest = hashlib.sha256()
    for part in (chunker_version.encode(), embedding_model.encode(), data):
        # Length prefix: ("ab", "c") and ("a", "bc") must not hash the same.
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()
