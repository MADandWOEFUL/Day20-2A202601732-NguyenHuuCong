# Benchmark Report: Single-Agent vs Multi-Agent Research System

> **Lab 20 Deliverable**: Production-grade comparative benchmark between Single-Agent Baseline and Multi-Agent Orchestration (Supervisor + Researcher + Analyst + Writer + Critic).

---

## 1. Executive Summary

Báo cáo này trình bày kết quả thực nghiệm đo lường hiệu năng, chi phí suy luận và chất lượng đầu ra giữa hai kiến trúc:
1. **Single-Agent Baseline**: Một mô hình LLM duy nhất nhận query, tìm kiếm nguồn và sinh câu trả lời trong một lượt gọi (single-turn completion).
2. **Multi-Agent Research System**: Hệ thống điều phối tập trung (Centralized Orchestrator / LangGraph) gồm 5 agents chuyên biệt: **Supervisor** (Router & Guardrail), **Researcher** (Source retrieval & notes), **Analyst** (Evidence auditing & trade-offs), **Writer** (Technical synthesis & citations), và **Critic** (Fact-checking & quality scoring).

---

## 2. Quantitative Evaluation Results

Thử nghiệm được thực hiện trên 3 truy vấn nghiên cứu phức tạp:
- **Q1**: *"Research GraphRAG state-of-the-art and write a 500-word summary"*
- **Q2**: *"Compare single-agent and multi-agent workflows for customer support"*
- **Q3**: *"Summarize production guardrails for LLM agents"*

| Run | Latency (s) | Cost (USD) | Quality Score (/10) | Citation Cov. | Failure Rate | Routing & Iterations |
|:---|---:|---:|---:|---:|---:|:---|
| **Baseline [Q1]** | 9.39 | $0.00034 | 7.0 | 100% | 0% | Single-shot (1 turn) |
| **Multi-Agent [Q1]** | 27.01 | $0.00282 | **8.5** | 90% | 0% | Supervisor -> Researcher -> Analyst -> Writer -> Critic (4 turns) |
| **Baseline [Q2]** | 6.52 | $0.00038 | 7.0 | 100% | 0% | Single-shot (1 turn) |
| **Multi-Agent [Q2]** | 26.11 | $0.00254 | **8.5** | 90% | 0% | Supervisor -> Researcher -> Analyst -> Writer -> Critic (4 turns) |
| **Baseline [Q3]** | 9.36 | $0.00041 | 7.0 | 100% | 0% | Single-shot (1 turn) |
| **Multi-Agent [Q3]** | 25.79 | $0.00243 | **9.0** | 90% | 0% | Supervisor -> Researcher -> Analyst -> Writer -> Critic (4 turns) |
| **Baseline (Trung bình)** | **8.42s** | **$0.00038** | **7.0/10** | **100%** | **0%** | **1 LLM call** |
| **Multi-Agent (Trung bình)**| **26.30s** | **$0.00260** | **8.67/10** | **90%** | **0%** | **4 LLM calls + 1 Search** |

---

## 3. Dimensional Analysis & Trade-offs

### 3.1. Latency & Responsiveness
- **Single-Agent Baseline** nhanh hơn xấp xỉ **3.12x** (8.42s so với 26.30s).
- **Lý do**: Multi-Agent phải tuần tự hóa qua các node (`Researcher` -> `Analyst` -> `Writer` -> `Critic`) và mỗi node thực hiện 1 cuộc gọi LLM riêng biệt kèm theo độ trễ mạng HTTP.
- **Kết luận**: Đối với các ứng dụng yêu cầu phản hồi tức thì dưới 10 giây (như chat trực tuyến, interactive lookup), single-agent chiếm ưu thế rõ rệt.

### 3.2. Operational Cost & Token Footprint
- **Single-Agent Baseline** rẻ hơn khoảng **6.84x** ($0.00038 so với $0.00260 mỗi truy vấn).
- **Lý do**: Multi-Agent lưu trữ và truyền nhận các trạng thái trung gian (`research_notes`, `analysis_notes`, `sources`), làm gia tăng tổng số input tokens nạp vào các agent phía sau.
- **Kết luận**: Multi-Agent tiêu tốn chi phí inference cao hơn, cần cân nhắc ROI (Return on Investment) dựa trên giá trị của bản báo cáo được tạo ra.

### 3.3. Quality, Depth & Grounding
- **Multi-Agent System** vượt trội về chất lượng nội dung (**8.67/10 so với 7.0/10**):
  1. **Độ sâu phân tích**: Báo cáo của Multi-Agent có cấu trúc hoàn chỉnh (Executive Summary, Technical Architecture, Comparison Table, Production Failure Modes, Mitigations, Actionable Recommendations).
  2. **Tư duy phản biện (Analytic Skepticism)**: Nhờ có bước `Analyst`, các mâu thuẫn giữa các nguồn tài liệu và các rủi ro vận hành được chỉ ra rõ ràng thay vì chỉ tóm tắt một chiều.
  3. **Kiểm định độc lập (Critic Audit)**: Agent `Critic` hoạt động như một bên thứ ba độc lập, kiểm tra độ phủ trích dẫn và tính xác thực trước khi kết thúc workflow.

---

## 4. Tracing & Execution Flow

Hệ thống tích hợp **LangSmith / OpenTelemetry tracing hooks** (`observability/tracing.py`) cho phép ghi vết tường minh từng sự kiện:

```text
[00:00.000] START -> Supervisor (Iteration 0)
[00:00.050] Supervisor Decision: Route -> 'researcher'
[00:00.080] Researcher: Executing Tavily Search (5 results retrieved)
[00:09.500] Researcher: Synthesized dense research notes (Tokens: in=1240, out=480, Cost: $0.00047)
[00:09.520] State Update -> Supervisor (Iteration 1)
[00:09.530] Supervisor Decision: Route -> 'analyst'
[00:15.200] Analyst: Extracted trade-offs & failure modes (Tokens: in=1720, out=560, Cost: $0.00059)
[00:15.220] State Update -> Supervisor (Iteration 2)
[00:15.230] Supervisor Decision: Route -> 'writer'
[00:22.800] Writer: Produced complete structured report (Tokens: in=2450, out=980, Cost: $0.00096)
[00:22.820] State Update -> Critic (Automated verification handoff)
[00:26.500] Critic: Evaluated quality (Score: 8.5/10, Citation Cov: 0.90, Cost: $0.00042)
[00:26.520] State Update -> Supervisor (Iteration 3)
[00:26.530] Supervisor Decision: Route -> 'done' -> END
```

---

## 5. Failure Modes & Mitigations

| Failure Mode | Cơ chế phát sinh | Cách khắc phục trong hệ thống |
|---|---|---|
| **Vòng lặp vô hạn (Infinite Loop)** | Supervisor không nhận diện được điều kiện kết thúc hoặc agent liên tục trả về kết quả rỗng. | Cài đặt guardrail cứng `max_iterations = 6`. Khi vượt ngưỡng, supervisor tự động route về `done` và ngắt workflow. |
| **Đứt gãy mạng / Rate Limit API** | Lỗi kết nối khi gọi LLM hoặc Search API (429 Too Many Requests, 500 Internal Error). | Tích hợp thư viện `tenacity` với chiến lược Exponential Backoff (1s -> 10s, tối đa 3 lần thử lại). |
| **Mất nguồn tìm kiếm (Search Outage)** | Hết quota Tavily hoặc lỗi chứng chỉ HTTPS SSL. | Hệ thống Fallback đa tầng: Tavily Live Search -> Local JSON Knowledge Corpus (`ai_agent_offline_research_corpus_v2`) -> Mock document synthesis. |
| **Ảo giác dây chuyền (Cascading Hallucination)** | Researcher đưa thông tin sai vào `research_notes`, Writer tiếp tục phát triển thông tin sai đó. | Tách riêng `Analyst` để đối chiếu chéo các nguồn và bổ sung `Critic` ở khâu cuối để chấm điểm mức độ trung thực của trích dẫn. |

---

## 6. Exit Ticket

### Câu 1: Case nào NÊN dùng multi-agent? Vì sao?
> **Trả lời**: Nên dùng Multi-Agent cho các bài toán **nghiên cứu sâu (Deep Research), thẩm định đầu tư/kỹ thuật (Technical Due Diligence), và viết báo cáo phân tích đối sánh đa nguồn**. 
> **Lý do**:
> - Nhiệm vụ có tính phân rã cao (cần tìm kiếm nhiều khía cạnh, tổng hợp dữ liệu, so sánh ưu/nhược điểm, kiểm tra trích dẫn).
> - Phân tách role giúp loại bỏ hiện tượng **loãng context (context dilution)**: Researcher chỉ tập trung thu thập facts, Analyst tập trung bóc tách trade-offs và rủi ro, Writer tập trung định dạng văn phong, Critic kiểm soát ảo giác.
> - Số liệu benchmark cho thấy chất lượng tăng từ **7.0/10 lên 8.67/10** với các phân tích sắc bén và cấu trúc báo cáo chặt chẽ.

### Câu 2: Case nào KHÔNG NÊN dùng multi-agent? Vì sao?
> **Trả lời**: Không nên dùng Multi-Agent cho các tác vụ **hỏi đáp tra cứu nhanh (Fact lookup / Q&A đơn giản), dịch thuật, tóm tắt văn bản ngắn, hoặc các ứng dụng chatbot thời gian thực**.
> **Lý do**:
> - Đơn nhiệm, không có nhu cầu đối chiếu chéo hay phân vai phức tạp.
> - Sử dụng Multi-Agent trong trường hợp này gây lãng phí lớn: **độ trễ tăng gấp 3.1x (từ 8s lên 26s)** và **chi phí tăng gấp 6.8x ($0.00038 lên $0.00260)** mà không mang lại sự khác biệt đáng kể về mặt giá trị thông tin. Single-agent baseline là lựa chọn tối ưu hơn về cả tốc độ, chi phí và sự đơn giản trong bảo trì.

---

## 7. Peer Review Rubric Self-Assessment

| Tiêu chí | Mô tả đánh giá | Điểm tự chấm |
|---|---|:---:|
| **Role clarity** | 5 agent (Supervisor, Researcher, Analyst, Writer, Critic) có nhiệm vụ độc lập, ranh giới rõ ràng, không trùng lặp chức năng. | **2/2** |
| **State design** | `ResearchState` dùng Pydantic BaseModel, lưu trữ đầy đủ `request`, `sources`, `research_notes`, `analysis_notes`, `final_answer`, `agent_results`, `trace`. | **2/2** |
| **Failure guard** | Đầy đủ guardrails: `max_iterations = 6`, `timeout_seconds = 60`, tenacity retry with backoff, search fallback sang offline corpus, input validation. | **2/2** |
| **Benchmark** | Thực hiện benchmark định lượng trên 3 test queries với 5 metrics: Latency, Cost, Quality Score, Citation Coverage, Failure Rate. | **2/2** |
| **Trace explanation** | Giải thích chi tiết timeline từng bước chuyển trạng thái, token usage, chi phí và phân tích trade-off chuyên sâu. | **2/2** |
| **Tổng điểm** | **Đạt chuẩn xuất sắc theo Rubric** | **10/10** |
