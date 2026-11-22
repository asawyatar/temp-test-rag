"""Deciding what one ingestion run must do.

Pure logic: no database, no files, no network. The ingestion job gathers two
snapshots for a single tenant, asks `plan_sync` what differs, then carries the
plan out. Keeping the decision separate from the doing is what makes
"re-running changes nothing" and "removing a source removes its chunks" easy
to test.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncPlan:
    """Source keys sorted into what the ingestion run must do with each."""

    to_add: frozenset[str]
    to_update: frozenset[str]
    to_delete: frozenset[str]
    unchanged: frozenset[str]

    @property
    def is_noop(self) -> bool:
        """True when applying this plan would not touch the database."""
        return not (self.to_add or self.to_update or self.to_delete)


def plan_sync(stored: Mapping[str, str], incoming: Mapping[str, str]) -> SyncPlan:
    """Compare what the database holds with what the input folder holds.

    Both arguments map a source key (the file's path relative to the tenant's
    folder, e.g. "rules/pool.ar.md") to a fingerprint. The fingerprint is a
    hash of the file's bytes plus the chunker version and embedding model name,
    so changing how we chunk or embed also counts as "changed".

    `stored` is what is in the `sources` table for this tenant right now.
    `incoming` is what the loader just found on disk.

    Every key from either mapping lands in exactly one of the four sets.

    An empty `incoming` yields a plan that deletes everything stored. This
    function reports what the folder says; deciding whether that is a mistake
    (a mistyped --path) is the caller's job, and `atar-ingest` refuses such a
    plan unless --allow-delete-all is given.
    """
    on_disk = incoming.keys()
    in_database = stored.keys()
    in_both = on_disk & in_database
    changed = frozenset(key for key in in_both if stored[key] != incoming[key])
    return SyncPlan(
        to_add=frozenset(on_disk - in_database),
        to_update=changed,
        to_delete=frozenset(in_database - on_disk),
        unchanged=frozenset(in_both) - changed,
    )
