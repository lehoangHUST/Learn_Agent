# jev-agent-lab

Mini lab: dùng **Jev** (TypeSafe AI, "System One model") như một *thành phần ra quyết định* bên trong AI agent Python — không phải như một agent.

> API và SDK trong code đã được đối chiếu với tài liệu chính thức (docs.typesafe.ai, OpenAPI `/v1/systemone`) và chạy thử với `typesafe-sdk==0.7.5` ngày 2026-10-21. Phần **offline** dùng provider giả (`examples/offline_fake.py`) — xác suất của nó là bịa, **không dùng để benchmark**.

## Cấu trúc

```
jev_lab/
  decisions.py      Domain types: ChoiceSignal, ScoreSignal, NoulSignal, DecisionResult (không import SDK)
  config.py         Mọi threshold + cost model ở một chỗ; threshold_from_costs()
  providers.py      DecisionProvider protocol, JevDecisionProvider (timeout, retry, phân loại lỗi,
                    sanity check response, log provenance), ScriptedProvider cho test
  fallback.py       RuleFallbackProvider + decide_with_fallback()
  ticket_routing.py Ví dụ A: định tuyến ticket (logic quyết định là hàm thuần decide_route)
  rag_gate.py       Ví dụ B: relevance gate / evidence sufficiency / claim validation quanh 1 LLM
  evaluation.py     ECE, Brier, log loss, reliability bins, risk-coverage/AURC, threshold theo cost,
                    negation coherence, bootstrap CI (pure Python)
examples/
  run_ticket_demo.py   python -m examples.run_ticket_demo [--live]
  run_rag_demo.py      python -m examples.run_rag_demo [--live]
  eval_tickets.py      python -m examples.eval_tickets data/tickets_labeled.jsonl [--live]
tests/                 30 unit test, không cần API key (python -m pytest -q)
```

## Chạy

```bash
pip install -e ".[dev]"
python -m pytest -q
python -m examples.run_ticket_demo                 # offline
export TYPESAFE_API_KEY=...                        # console.typesafe.ai
python -m examples.run_ticket_demo --live          # gọi Jev thật
python -m examples.eval_tickets data/tickets_labeled.jsonl --live
```

`JEV_MODEL` mặc định là `jev-1.13.0` (pin version, không dùng alias `jev-latest`, vì threshold được tune theo một version cụ thể).

## Nguyên tắc thiết kế

1. **Model đưa tín hiệu, code ra quyết định.** `decide_route()` và `route_passage()` là hàm thuần nhận xác suất, trả hành động. Threshold, business rule, quyền hạn đều nằm trong code và `config.py`.
2. **Threshold khác nhau cho mỗi quyết định**, suy ra từ chi phí sai (`threshold_from_costs`) và được tune lại trên dữ liệu của bạn.
3. **Cổng rủi ro cao dùng ngưỡng thấp (high recall)** và luôn qua người; không bao giờ để xác suất là lớp bảo vệ duy nhất.
4. **Fallback rõ ràng**: Jev lỗi/timeout → rule-based → review queue; RAG → chế độ `degraded_ungated` được gắn nhãn.
5. **Không log `state`** (thường chứa PII); log model version, request id, latency, token.

## Hai bẫy đã gặp khi viết code (đã có test)

* `TypeSafeAPITimeoutError` kế thừa `TimeoutError`, mà trên Python ≥ 3.11 `asyncio.TimeoutError is TimeoutError` → phải bắt lỗi SDK *trước* `asyncio.TimeoutError`.
* `SystemOneResponse.request_id` là property **ném `TypeSafeError`** khi thiếu header `x-typesafe-request-id` (có thể xảy ra qua gateway) → bọc try.
* Với Score, SDK parse key `"0","1",...` thành `int`.
* Fallback không được trả 0.5 ("không biết") cho Noul: nó sẽ kích hoạt mọi cổng high-recall ngưỡng thấp.

## Đánh giá trên dữ liệu thật

`eval_tickets.py` cache mọi response theo hash request (chạy lại miễn phí), chia train/test **trước khi** tune threshold, và báo cáo: accuracy, ECE, Brier, log loss, reliability bins, accuracy@coverage, AURC, tỷ lệ đổi đáp án khi đảo thứ tự option, threshold security tune theo cost trên train và đo trên test, latency p50/p95, tỷ lệ fallback.

20 dòng trong `data/tickets_labeled.jsonl` chỉ để chạy thử pipeline. Cần vài trăm đến vài nghìn mẫu có nhãn từ hệ thống thật (gồm tiếng Việt nếu traffic của bạn là tiếng Việt — TypeSafe nói tiếng Anh là ngôn ngữ huấn luyện chính) trước khi tin bất kỳ con số nào.
