"""Test tầng HTTP.

Điểm đặc thù của harness khi đưa lên HTTP: ``POST /ask`` **không** luôn trả về câu
trả lời. Nếu agent xin một hành động cần phê duyệt thì request kết thúc với
``202`` và episode vẫn sống trong checkpoint; một request khác gửi quyết định tới
``/approve`` để chạy tiếp. Toàn bộ phần đó được kiểm ở đây.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api as api_module
from app.api import app, get_runner, get_settings
from harness.model import call, calls, say
from toolsets.sql.tools import FindingStore


@pytest.fixture
def client(settings, make_runner, request):
    """TestClient với runner đã tiêm kịch bản.

    Không dùng ``with TestClient(...)`` để lifespan không chạy: lifespan sẽ dựng
    dependency thật và cần OPENAI_API_KEY.
    """
    script = getattr(request, "param", None) or [say("xong")]
    runner = make_runner(script)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_runner] = lambda: runner
    api_module._state["settings"] = settings
    api_module._state["runner"] = runner
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()
        api_module._state.clear()


# --------------------------------------------------------------------------- #
# /ask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "client",
    [[calls(call("list_tables")), say("Database có 4 bảng và 1 view.")]],
    indirect=True,
)
def test_ask_returns_the_answer(client: TestClient) -> None:
    response = client.post("/ask", json={"goal": "database có gì?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert "4 bảng" in body["final_answer"]
    assert body["thread_id"].startswith("th_")
    assert body["usage"]["tool_calls_total"] == 1
    assert body["pending_approvals"] == []
    # Transcript trả ra cho client: với harness, "agent đã làm gì" là thông tin
    # người dùng cần, không phải chi tiết nội bộ.
    assert any(step["kind"] == "tool" for step in body["steps"])


def test_ask_rejects_empty_goal(client: TestClient) -> None:
    assert client.post("/ask", json={"goal": "x"}).status_code == 422


@pytest.mark.parametrize("client", [[calls(call("list_tables"))] * 20], indirect=True)
def test_ask_reports_budget_stop_as_200_not_an_error(client: TestClient) -> None:
    """Hết hạn mức là kết thúc bình thường: 200 với ``status='stopped'``.

    Trả 500 ở đây sẽ sai: không có gì hỏng, agent chỉ không hoàn thành trong hạn mức,
    và những gì đã thu được vẫn dùng được.
    """
    response = client.post("/ask", json={"goal": "lặp mãi"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stopped"
    assert body["stop_reason"] == "budget_max_model_calls"
    assert "Harness đã dừng" in body["final_answer"]


# --------------------------------------------------------------------------- #
# Vòng phê duyệt qua HTTP
# --------------------------------------------------------------------------- #
APPROVAL_SCRIPT = [
    calls(
        call(
            "save_finding",
            call_id="w1",
            title="Doanh thu tập trung ở đơn paid",
            summary="Theo view order_revenue, phần lớn doanh thu từ đơn trạng thái paid.",
            sql="SELECT status, SUM(revenue) FROM order_revenue GROUP BY status",
        )
    ),
    say("Đã xử lý yêu cầu lưu finding."),
]


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_ask_returns_202_when_approval_is_needed(client: TestClient, findings_path: Path) -> None:
    response = client.post("/ask", json={"goal": "lưu kết luận"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "awaiting_approval"
    assert len(body["pending_approvals"]) == 1
    pending = body["pending_approvals"][0]
    assert pending["tool_call_id"] == "w1"
    assert pending["tool_name"] == "save_finding"
    assert pending["risk"]
    assert "title" in pending["arguments"]
    # Chưa có gì được ghi.
    assert not findings_path.exists()


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_approve_resumes_and_writes(client: TestClient, findings_path: Path) -> None:
    paused = client.post("/ask", json={"goal": "lưu kết luận"}).json()

    response = client.post(
        f"/episodes/{paused['thread_id']}/approve",
        json={"decisions": {"w1": {"approved": True, "reason": "hợp lý"}}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["usage"]["tool_calls_by_name"]["save_finding"] == 1
    stored = json.loads(findings_path.read_text(encoding="utf-8"))
    assert len(stored) == 1


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_deny_resumes_without_writing(client: TestClient, findings_path: Path) -> None:
    paused = client.post("/ask", json={"goal": "lưu kết luận"}).json()

    body = client.post(
        f"/episodes/{paused['thread_id']}/approve",
        json={"decisions": {"w1": {"approved": False, "reason": "chưa đủ căn cứ"}}},
    ).json()

    assert body["status"] == "answered"
    assert FindingStore(findings_path).count() == 0
    denied = [s for s in body["steps"] if s["kind"] == "tool" and s["status"] == "denied"]
    assert len(denied) == 1


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_empty_decisions_body_denies(client: TestClient, findings_path: Path) -> None:
    """Body không có quyết định nào = từ chối, không phải treo và không phải đồng ý."""
    paused = client.post("/ask", json={"goal": "lưu kết luận"}).json()

    body = client.post(f"/episodes/{paused['thread_id']}/approve", json={}).json()

    assert body["status"] == "answered"
    assert FindingStore(findings_path).count() == 0


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_decision_for_a_different_tool_call_id_denies(
    client: TestClient, findings_path: Path
) -> None:
    """Đồng ý một id khác không mở được id đang chờ."""
    paused = client.post("/ask", json={"goal": "lưu kết luận"}).json()

    client.post(
        f"/episodes/{paused['thread_id']}/approve",
        json={"decisions": {"id-khac": {"approved": True}}},
    )

    assert FindingStore(findings_path).count() == 0


@pytest.mark.parametrize("client", [APPROVAL_SCRIPT], indirect=True)
def test_episode_state_endpoint(client: TestClient) -> None:
    paused = client.post("/ask", json={"goal": "lưu kết luận"}).json()

    during = client.get(f"/episodes/{paused['thread_id']}").json()
    assert during["exists"] is True
    # next không rỗng = episode đang dừng giữa đường.
    assert during["next_nodes"] == ["execute_tools"]

    client.post(
        f"/episodes/{paused['thread_id']}/approve",
        json={"decisions": {"w1": {"approved": True}}},
    )
    after = client.get(f"/episodes/{paused['thread_id']}").json()
    assert after["next_nodes"] == []
    assert after["status"] == "answered"


def test_unknown_thread_is_reported_not_crashed(client: TestClient) -> None:
    body = client.get("/episodes/th_khong-ton-tai").json()
    assert body["exists"] is False


# --------------------------------------------------------------------------- #
# /tools, /health
# --------------------------------------------------------------------------- #
def test_tools_endpoint_exposes_risk_labels(client: TestClient) -> None:
    """Người dùng harness có quyền biết agent *có thể* làm gì và việc nào cần họ duyệt."""
    body = client.get("/tools").json()
    by_name = {tool["name"]: tool for tool in body["tools"]}

    assert by_name["run_query"]["requires_approval"] is False
    assert by_name["save_finding"]["requires_approval"] is True
    assert by_name["save_finding"]["side_effect"] == "write"
    assert by_name["save_finding"]["allowed_for_you"] is True
    assert body["budget"]["max_model_calls"] == 6


def test_tools_endpoint_marks_tools_without_permission(client: TestClient, settings) -> None:
    import dataclasses

    from app.api import get_settings as settings_dep

    app.dependency_overrides[settings_dep] = lambda: dataclasses.replace(
        settings, permissions=frozenset({"sql:read"})
    )
    body = client.get("/tools").json()
    by_name = {tool["name"]: tool for tool in body["tools"]}
    assert by_name["save_finding"]["allowed_for_you"] is False
    assert by_name["run_query"]["allowed_for_you"] is True


def test_health_reports_missing_api_key_as_degraded(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert any("OPENAI_API_KEY" in w for w in body["warnings"])
    assert body["database"]["exists"] is True


def test_health_does_not_leak_secrets(client: TestClient, settings) -> None:
    import dataclasses

    from app.api import get_settings as settings_dep

    app.dependency_overrides[settings_dep] = lambda: dataclasses.replace(
        settings, openai_api_key="sk-rat-la-bi-mat"
    )
    text = client.get("/health").text
    assert "sk-rat-la-bi-mat" not in text
    assert "***set***" in text


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #
def test_openapi_documents_the_202_behaviour(client: TestClient) -> None:
    """Hành vi 202 là thứ dễ gây bất ngờ nhất cho người tích hợp -> phải ghi rõ."""
    spec = client.get("/openapi.json").json()
    assert "202" in spec["info"]["description"]
    assert "pending_approvals" in spec["info"]["description"]
    assert "202" in spec["paths"]["/ask"]["post"]["responses"]
