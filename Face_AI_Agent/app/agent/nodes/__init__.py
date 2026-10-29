"""Các node của graph.

Mỗi node là một hàm ``(state, runtime) -> partial update``. Không node nào import
FastAPI, và không node nào tự dựng service - dependency đến qua
``runtime.context.deps``. Nhờ vậy test gọi node trực tiếp được.
"""

from app.agent.nodes.analyze import analyze_results
from app.agent.nodes.answer import answer_question, render_template_answer
from app.agent.nodes.begin import begin_run
from app.agent.nodes.detect import detect_faces
from app.agent.nodes.embed import generate_embeddings
from app.agent.nodes.error import handle_error
from app.agent.nodes.finalize import finalize_response
from app.agent.nodes.intent import classify_by_rules, classify_intent
from app.agent.nodes.plan import plan_execution
from app.agent.nodes.search import search_faces
from app.agent.nodes.validate import validate_input

__all__ = [
    "analyze_results",
    "answer_question",
    "begin_run",
    "classify_by_rules",
    "classify_intent",
    "detect_faces",
    "finalize_response",
    "generate_embeddings",
    "handle_error",
    "plan_execution",
    "render_template_answer",
    "search_faces",
    "validate_input",
]
