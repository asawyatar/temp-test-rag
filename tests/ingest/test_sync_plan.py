from atar_ai_agent.ingest.sync import plan_sync


def test_source_only_on_disk_is_added() -> None:
    plan = plan_sync(stored={}, incoming={"pool.md": "h1"})

    assert plan.to_add == {"pool.md"}
    assert not (plan.to_update or plan.to_delete or plan.unchanged)


def test_source_with_same_fingerprint_is_left_alone() -> None:
    plan = plan_sync(stored={"pool.md": "h1"}, incoming={"pool.md": "h1"})

    assert plan.unchanged == {"pool.md"}
    assert plan.is_noop


def test_source_with_different_fingerprint_is_updated() -> None:
    plan = plan_sync(stored={"pool.md": "h1"}, incoming={"pool.md": "h2"})

    assert plan.to_update == {"pool.md"}
    assert not plan.is_noop


def test_source_missing_from_disk_is_deleted() -> None:
    plan = plan_sync(
        stored={"pool.md": "h1", "fees.md": "h1"},
        incoming={"pool.md": "h1"},
    )

    assert plan.to_delete == {"fees.md"}
    assert plan.unchanged == {"pool.md"}


def test_every_source_lands_in_exactly_one_bucket() -> None:
    stored = {"same.md": "h1", "edited.md": "h1", "removed.md": "h1"}
    incoming = {"same.md": "h1", "edited.md": "h2", "new.md": "h1"}

    plan = plan_sync(stored, incoming)

    assert plan.to_add == {"new.md"}
    assert plan.to_update == {"edited.md"}
    assert plan.to_delete == {"removed.md"}
    assert plan.unchanged == {"same.md"}


def test_rerunning_on_the_same_input_is_a_noop() -> None:
    """Idempotency: once a folder is ingested, `stored` equals `incoming`."""
    folder = {"pool.md": "h1", "fees.md": "h2"}

    assert plan_sync(stored=folder, incoming=folder).is_noop


def test_empty_folder_plans_to_delete_everything_stored() -> None:
    """Reported honestly; atar-ingest refuses it without --allow-delete-all."""
    plan = plan_sync(stored={"pool.md": "h1", "fees.md": "h2"}, incoming={})

    assert plan.to_delete == {"pool.md", "fees.md"}
    assert not (plan.to_add or plan.to_update or plan.unchanged)


def test_nothing_anywhere_is_a_noop() -> None:
    assert plan_sync(stored={}, incoming={}).is_noop
