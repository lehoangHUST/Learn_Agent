"""HTTP API - nơi vòng lặp phê duyệt trở thành hai request.

Điểm đáng chú ý của một harness khi đưa lên HTTP: ``POST /ask`` **không** luôn trả
về câu trả lời. Nếu agent xin một hành động cần phê duyệt, request đó kết thúc với
``202`` và danh sách việc cần duyệt; episode vẫn sống trong checkpoint. Một request
khác - có thể từ người khác, nhiều giờ sau - gửi quyết định tới
``POST /episodes/{thread_id}/approve`` và episode chạy tiếp từ đúng chỗ đã dừng.

Đó là lý do ``thread_id`` phải trả về cho client, và là lý do checkpointer không
phải tuỳ chọn khi có tool ghi dữ liệu.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import Settings, load_settings
from app.wiring import build_runner
from harness.errors import HarnessError
from harness.runner import HarnessRunner
from harness.transcript import summarize

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level.upper())
    _state["settings"] = settings
    # Dựng một lần: compile graph kiểm tra topology, và SqliteSaver giữ một
    # connection dùng lại.
    _state["runner"] = build_runner(settings)
    logger.info("api_started", extra=settings.redacted())
    try:
        yield
    finally:
        _state.clear()


def get_settings() -> Settings:
    return _state.get("settings") or load_settings()


def get_runner() -> HarnessRunner:
    runner = _state.get("runner")
    if runner is None:  # pragma: no cover - chỉ xảy ra nếu gọi ngoài lifespan
        runner = build_runner(get_settings())
        _state["runner"] = runner
    return runner


SettingsDep = Annotated[Settings, Depends(get_settings)]
RunnerDep = Annotated[HarnessRunner, Depends(get_runner)]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=2000, description="câu hỏi về dữ liệu")
    user_id: str = Field(default="api-user")
    thread_id: str | None = Field(
        default=None, description="truyền lại để tiếp tục trong cùng một thread"
    )


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


class ApproveRequest(BaseModel):
    user_id: str = Field(default="api-user")
    # Theo từng tool_call_id: một lượt có thể xin hai hành động ghi và người duyệt
    # có quyền đồng ý cái này, từ chối cái kia.
    decisions: dict[str, ApprovalDecision] = Field(
        default_factory=dict,
        description="map tool_call_id -> quyết định. Thiếu một id = TỪ CHỐI id đó.",
    )


class PendingItem(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    risk: str
    reason: str = ""


class EpisodeResponse(BaseModel):
    episode_id: str
    thread_id: str
    status: Literal["answered", "stopped", "awaiting_approval", "failed", "unknown"]
    stop_reason: str | None = None
    final_answer: str = ""
    pending_approvals: list[PendingItem] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] = Field(default_factory=list)


def _to_response(result: Any) -> EpisodeResponse:
    return EpisodeResponse(
        episode_id=result.episode_id,
        thread_id=result.thread_id,
        status=result.status,  # type: ignore[arg-type]
        stop_reason=result.output.get("stop_reason"),
        final_answer=result.final_answer,
        pending_approvals=[PendingItem(**item) for item in result.pending],
        usage=summarize(result.output),
        steps=list(result.output.get("steps") or []),
    )


# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Agent Harness - SQL analyst",
    version="0.1.0",
    description=(
        "Harness cho một agent phân tích dữ liệu: **LLM là bộ điều khiển**, harness "
        "cấp tool, hạn mức, phân quyền và cổng phê duyệt.\n\n"
        "- `POST /ask` có thể trả về **202** kèm `pending_approvals` thay vì câu trả "
        "lời: agent đã xin một hành động cần người đồng ý.\n"
        "- Gửi quyết định tới `POST /episodes/{thread_id}/approve` để episode chạy tiếp.\n"
        "- Thiếu quyết định cho một `tool_call_id` nghĩa là **từ chối** id đó."
    ),
    lifespan=lifespan,
)


@app.exception_handler(HarnessError)
async def harness_error_handler(request: Request, exc: HarnessError) -> JSONResponse:
    """``HarnessError`` là lỗi cấu hình/bất biến, không phải lỗi của model.

    Lỗi của model (gọi sai tool, SQL sai) không bao giờ tới đây - chúng đã thành
    ``ToolMessage`` trong episode và nằm trong transcript.
    """
    logger.warning("harness_error", extra={"code": exc.code, "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": {"code": exc.code, "message": str(exc)}},
    )


@app.post(
    "/ask",
    response_model=EpisodeResponse,
    responses={202: {"model": EpisodeResponse, "description": "đang chờ phê duyệt"}},
    summary="Đặt một câu hỏi về dữ liệu",
)
def ask(body: AskRequest, runner: RunnerDep) -> Any:
    result = runner.start(body.goal, user_id=body.user_id, thread_id=body.thread_id)
    payload = _to_response(result)
    if result.awaiting_approval:
        # 202 chứ không 200: việc chưa xong, client phải hành động tiếp.
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=payload.model_dump(mode="json"),
        )
    return payload


@app.post(
    "/episodes/{thread_id}/approve",
    response_model=EpisodeResponse,
    responses={202: {"model": EpisodeResponse, "description": "còn yêu cầu phê duyệt khác"}},
    summary="Gửi quyết định phê duyệt và tiếp tục episode",
)
def approve(thread_id: str, body: ApproveRequest, runner: RunnerDep) -> Any:
    decisions = {
        call_id: decision.model_dump() for call_id, decision in body.decisions.items()
    }
    result = runner.resume(thread_id, decisions, user_id=body.user_id)
    payload = _to_response(result)
    if result.awaiting_approval:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=payload.model_dump(mode="json"),
        )
    return payload


@app.get("/episodes/{thread_id}", summary="Trạng thái hiện tại của một episode")
def get_episode(thread_id: str, runner: RunnerDep) -> dict[str, Any]:
    snapshot = runner.get_state(thread_id)
    values = snapshot.values or {}
    return {
        "thread_id": thread_id,
        "exists": bool(values),
        "status": values.get("status"),
        "stop_reason": values.get("stop_reason"),
        "final_answer": values.get("final_answer", ""),
        # next rỗng nghĩa là episode đã chạy xong; có giá trị nghĩa là đang dừng
        # giữa đường (thường là chờ phê duyệt).
        "next_nodes": list(snapshot.next or ()),
        "usage": summarize(values) if values else {},
    }


@app.get("/tools", summary="Tool mà agent được phép dùng, kèm nhãn rủi ro")
def list_tools(settings: SettingsDep, runner: RunnerDep) -> dict[str, Any]:
    """Công khai danh sách tool và nhãn rủi ro.

    Đây là thông tin vận hành, không phải chi tiết nội bộ: người dùng một harness có
    quyền biết agent *có thể* làm những gì, và hành động nào sẽ cần họ bấm đồng ý.
    """
    registry = runner.context_for("catalog-reader").registry
    return {
        "permissions_granted": sorted(settings.permissions),
        "budget": settings.redacted()["budget"],
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "side_effect": spec.side_effect,
                "requires_approval": spec.requires_approval,
                "required_permission": spec.required_permission,
                "risk": spec.risk,
                "allowed_for_you": (
                    spec.required_permission is None
                    or spec.required_permission in settings.permissions
                ),
            }
            for spec in registry.all()
        ],
    }


@app.get("/health", summary="Trạng thái dịch vụ")
def health(settings: SettingsDep) -> dict[str, Any]:
    db_exists = settings.db_path.exists()
    warnings: list[str] = []
    if not db_exists:
        warnings.append(f"không có database tại {settings.db_path}; chạy python -m toolsets.sql.seed")
    if settings.model_provider == "openai" and not settings.openai_api_key:
        warnings.append("thiếu OPENAI_API_KEY; /ask sẽ lỗi (dùng CLI --demo để chạy offline)")
    if settings.checkpointer == "none":
        warnings.append("checkpointer=none; tool cần phê duyệt sẽ không chạy được")
    return {
        "status": "ok" if not warnings else "degraded",
        "database": {"path": str(settings.db_path), "exists": db_exists},
        "config": settings.redacted(),
        "warnings": warnings,
    }
