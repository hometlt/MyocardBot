from datetime import UTC, datetime, timedelta

from miocard.support import SupportStore


def test_analysis_cases_persist_and_latest_is_selected(tmp_path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SupportStore(path)
    first = store.create_analysis_case(100, datetime(2026, 8, 27, 8, 0, tzinfo=UTC))
    second = store.create_analysis_case(
        100,
        datetime(2026, 8, 27, 8, 0, tzinfo=UTC) + timedelta(minutes=5),
    )
    store.update_analysis_outcome(second.case_id, "Готово")

    restored = SupportStore(path)

    assert restored.get_analysis_case(first.case_id) == first
    latest = restored.latest_analysis_case(100)
    assert latest is not None
    assert latest.case_id == second.case_id
    assert latest.outcome == "Готово"


def test_pending_support_is_consumed_explicitly(tmp_path) -> None:
    store = SupportStore(tmp_path / "support.sqlite3")
    case = store.create_analysis_case(100, datetime.now(UTC))
    store.set_pending(100, "result", case.case_id)

    assert store.get_pending(100) is not None
    pending = store.pop_pending(100)

    assert pending is not None
    assert pending.category == "result"
    assert pending.analysis_case_id == case.case_id
    assert store.get_pending(100) is None


def test_known_users_persist_and_deduplicate_across_restart(tmp_path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SupportStore(path)
    store.record_user_seen(100)
    store.record_user_seen(200)
    store.record_user_seen(100)

    restored = SupportStore(path)

    assert sorted(restored.list_known_user_ids()) == [100, 200]


def test_routes_preserve_exact_moderator_across_restart(tmp_path) -> None:
    path = tmp_path / "support.sqlite3"
    store = SupportStore(path)
    ticket_id = store.create_ticket(100, "other", None)
    store.add_route(
        chat_id=100,
        message_id=501,
        ticket_id=ticket_id,
        user_id=100,
        moderator_id=900,
    )

    route = SupportStore(path).get_route(100, 501)

    assert route is not None
    assert route.ticket_id == ticket_id
    assert route.user_id == 100
    assert route.moderator_id == 900

