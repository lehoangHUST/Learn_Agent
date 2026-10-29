"""Tool registry - mặt phẳng ghép giữa harness và một domain cụ thể.

Harness không biết gì về SQL, về khuôn mặt hay về bất cứ domain nào. Nó biết đúng
một thứ: một **tool** là bộ bốn gồm tên, schema tham số, handler, và *nhãn rủi ro*.

Nhãn rủi ro là phần mà một tool registry thông thường không có, nhưng harness thì
bắt buộc phải có, vì model là thứ chọn tool. Mỗi ``ToolSpec`` khai báo:

* ``side_effect`` - ``read`` hay ``write``. Thứ gì thay đổi thế giới thì không được
  chạy chỉ vì một model nói nên chạy.
* ``required_permission`` - quyền mà principal phải có. Kiểm ở harness, không ở
  trong handler: một tool mới viết sẽ không thể "quên" kiểm quyền.
* ``requires_approval`` - có cần người bấm đồng ý trước khi chạy.

Schema tham số dùng pydantic. Hai lợi ích, cả hai đều cần: sinh JSON Schema cho
model, và **validate** tham số mà model trả về - vì model trả sai kiểu là chuyện
thường, không phải ngoại lệ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from harness.errors import ArgumentInvalid, DuplicateTool, ToolNotRegistered

SIDE_EFFECT_READ = "read"
SIDE_EFFECT_WRITE = "write"


@dataclass(slots=True)
class ToolRuntime:
    """Thứ handler nhận ngoài tham số: danh tính người gọi và cấu hình domain.

    Handler **không** nhận state của harness. Lý do: handler không được phép đọc
    hay sửa tiến trình của episode - nó là một hàm thuần về mặt điều khiển, chỉ
    nhận tham số và trả về văn bản.
    """

    user_id: str
    context: dict[str, Any] = field(default_factory=dict)


# Handler nhận tham số đã validate và trả về **văn bản** cho model đọc.
# Trả về str chứ không phải object: mọi thứ model thấy đều là text, nên biên giới
# đó nên nằm ngay ở handler thay vì để harness tự đoán cách serialize.
ToolHandler = Callable[[Any, ToolRuntime], str]


class SupportsJsonSchema(Protocol):
    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    side_effect: str = SIDE_EFFECT_READ
    required_permission: str | None = None
    requires_approval: bool = False
    # Câu mô tả rủi ro, hiện ra cho người phê duyệt. Viết cho người đọc, không
    # phải cho model.
    risk: str = ""

    def __post_init__(self) -> None:
        if self.side_effect not in (SIDE_EFFECT_READ, SIDE_EFFECT_WRITE):
            raise ValueError(f"side_effect không hợp lệ: {self.side_effect!r}")
        if self.side_effect == SIDE_EFFECT_WRITE and not self.requires_approval:
            # Không phải quy ước mềm: một tool thay đổi thế giới mà không cần ai
            # đồng ý thì nhãn rủi ro trở thành vô nghĩa. Chặn ngay lúc đăng ký.
            raise ValueError(
                f"tool {self.name!r} có side_effect=write thì phải requires_approval=True"
            )

    def validate_arguments(self, raw: dict[str, Any]) -> BaseModel:
        """Validate tham số model sinh ra. Sai -> ``ArgumentInvalid`` để báo lại model."""
        if not isinstance(raw, dict):
            raise ArgumentInvalid(
                f"tham số của {self.name} phải là object, nhận được {type(raw).__name__}",
                details={"tool": self.name},
            )
        try:
            return self.args_model.model_validate(raw)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()[:5]
            ]
            raise ArgumentInvalid(
                f"tham số của {self.name} không đúng schema",
                hint="; ".join(problems) + f". Schema: {self.json_schema()}",
                details={"tool": self.name, "problems": problems},
            ) from exc

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def to_openai_tool(self) -> dict[str, Any]:
        """Định dạng tool của OpenAI - cũng là định dạng ``bind_tools`` nhận."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }

    def summary_line(self) -> str:
        """Một dòng cho system prompt. Nói rõ tool nào cần phê duyệt."""
        flags = []
        if self.side_effect == SIDE_EFFECT_WRITE:
            flags.append("GHI")
        if self.requires_approval:
            flags.append("CẦN PHÊ DUYỆT")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"- {self.name}{suffix}: {self.description}"


class ToolRegistry:
    """Tập tool mà model được phép nhìn thấy trong một episode."""

    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolRegistry:
        if spec.name in self._specs:
            raise DuplicateTool(f"tool {spec.name!r} đã được đăng ký")
        self._specs[spec.name] = spec
        return self

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolNotRegistered(f"tool {name!r} chưa đăng ký") from exc

    def has(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return sorted(self._specs)

    def all(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.names()]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self.all()]

    def catalog_text(self) -> str:
        return "\n".join(spec.summary_line() for spec in self.all())

    def __len__(self) -> int:
        return len(self._specs)
