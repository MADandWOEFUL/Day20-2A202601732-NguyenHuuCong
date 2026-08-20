# Design Document: Multi-Agent Research System

## Problem

Hệ thống **Multi-Agent Research Assistant** được thiết kế nhằm tự động hóa quy trình nghiên cứu kỹ thuật chuyên sâu (Deep Technical Research & Synthesis). Hệ thống tiếp nhận các câu hỏi nghiên cứu mở hoặc yêu cầu so sánh phức tạp, tự động thu thập tài liệu minh chứng (qua Live Web Search API hoặc Offline Knowledge Corpus), bóc tách và phân tích dữ liệu đa chiều, đánh giá độ tin cậy của bằng chứng và biên soạn báo cáo kỹ thuật hoàn chỉnh với đầy đủ trích dẫn (citations) có thể kiểm chứng.

## Why multi-agent?

1. **Khắc phục hiện tượng loãng ngữ cảnh (Context Dilution)**: Một mô hình LLM đơn lẻ khi phải đảm nhiệm đồng thời nhiều tác vụ (sinh query tìm kiếm, đọc hàng chục tài liệu, đối sánh luận điểm, kiểm định trích dẫn và định dạng văn bản) dễ bị quá tải context, bỏ sót dữ liệu quan trọng hoặc sinh ảo giác (hallucinations).
2. **Nguyên lý phân tách trách nhiệm (Separation of Concerns)**: Mỗi agent tập trung vào một năng lực cốt lõi:
   - Thu thập thông tin khách quan (Researcher).
   - Tư duy phản biện và phân tích rủi ro (Analyst).
   - Biên soạn mạch lạc theo văn phong kỹ thuật (Writer).
   - Kiểm định tính chính xác và trích dẫn (Critic).
3. **Khả năng quan sát và kiểm soát lỗi (Observability & Debuggability)**: Shared state cho phép trace lại chính xác agent nào đưa ra nhận định sai, bước nào bị nghẽn và chi phí token của từng mắt xích trong chuỗi xử lý.
4. **Áp dụng Guardrails chuẩn xác**: Dễ dàng áp dụng giới hạn số lần lặp (max iterations), timeout cho từng agent, fallback và retry độc lập.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode & Mitigation |
|---|---|---|---|---|
| **Supervisor** | Điều phối toàn cục, kiểm tra trạng thái shared state và quyết định agent tiếp theo; ngắt chu trình. | `ResearchState` hiện tại | Tên agent tiếp theo (`researcher`, `analyst`, `writer`, `done`) | **Lặp vô hạn (Infinite Loop)**: Khống chế bằng `max_iterations = 6`. |
| **Researcher** | Tìm kiếm tài liệu (Tavily/Corpus), trích lọc facts và tóm tắt thành ghi chú nghiên cứu. | `state.request`, tài liệu từ SearchClient | `state.sources`, `state.research_notes` | **Mất kết nối mạng/Không tìm thấy nguồn**: Fallback sang local offline corpus. |
| **Analyst** | Bóc tách luận điểm chính, so sánh trade-offs, đánh giá độ tin cậy bằng chứng và chỉ ra rủi ro/gaps. | `state.research_notes`, `state.sources` | `state.analysis_notes` | **Đánh giá thiên vị / thiếu chiều sâu**: Định hướng qua hệ thống prompt phân tích đa chiều. |
| **Writer** | Tổng hợp nghiên cứu và phân tích thành báo cáo hoàn chỉnh, chuẩn cấu trúc, kèm trích dẫn nguồn. | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | **Ảo giác trích dẫn / định dạng sai**: Buộc gắn source tags `[Source N]` tường minh. |
| **Critic** | Kiểm định tính xác thực, tính toán độ phủ trích dẫn và chấm điểm chất lượng báo cáo. | `state.final_answer`, `state.sources` | Audit metadata: `quality_score`, `citation_coverage`, feedback | **Không parse được JSON review**: Fallback metric mặc định và log cảnh báo. |

## Shared state

Cấu trúc dữ liệu trung tâm `ResearchState` kế thừa từ `pydantic.BaseModel`:

- `request: ResearchQuery`: Chứa câu hỏi nghiên cứu, giới hạn số nguồn (`max_sources`), và đối tượng độc giả (`audience`).
- `iteration: int`: Bộ đếm số lượt chuyển tiếp của supervisor, dùng làm cơ chế ngắt guardrail.
- `route_history: list[str]`: Nhật ký định tuyến ghi lại thứ tự các agent đã thực thi (`['researcher', 'analyst', 'writer', 'done']`).
- `sources: list[SourceDocument]`: Danh sách tài liệu thu thập được kèm metadata (tiêu đề, URL, snippet, score).
- `research_notes: str | None`: Ghi chú kỹ thuật thô từ Researcher.
- `analysis_notes: str | None`: Bản phân tích đối sánh, rủi ro và nhận định từ Analyst.
- `final_answer: str | None`: Báo cáo nghiên cứu kỹ thuật hoàn chỉnh từ Writer.
- `agent_results: list[AgentResult]`: Lưu vết kết quả, token input/output, và chi phí USD của từng agent.
- `trace: list[dict[str, Any]]`: Timeline các sự kiện (event-level spans) phục vụ tracing và debug.
- `errors: list[str]`: Danh sách lỗi (nếu có) phát sinh trong quá trình chạy.

## Routing policy

Hệ thống sử dụng mô hình **Centralized Supervisor / Hub-and-Spoke Architecture**:

```text
       ┌──────────────┐
       │    START     │
       └──────┬───────┘
              │
              ▼
    ┌──────────────────┐ ◄──────────────┐
    │    Supervisor    │                │
    └─────────┬────────┘                │
              │ (Conditional Route)     │
       ┌──────┼──────────────┬──────────┴────────┐
       ▼      ▼              ▼                   ▼
 ┌──────────┐ ┌─────────┐ ┌────────┐        ┌─────────┐
 │Researcher│ │ Analyst │ │ Writer │        │   END   │
 └────┬─────┘ └────┬────┘ └───┬────┘        └─────────┘
      │            │          │
      │            │          ▼
      │            │      ┌────────┐
      │            │      │ Critic │
      │            │      └───┬────┘
      └────────────┴──────────┘
```

1. **START** -> **Supervisor**: Khởi tạo phiên làm việc.
2. **Supervisor Routing Policy**:
   - Nếu `iteration >= max_iterations` -> Chuyển sang `END`.
   - Nếu chưa có `sources` & `research_notes` -> Điều hướng đến `Researcher`.
   - Nếu đã có `research_notes` nhưng chưa có `analysis_notes` -> Điều hướng đến `Analyst`.
   - Nếu đã có `analysis_notes` nhưng chưa có `final_answer` -> Điều hướng đến `Writer`.
   - Sau khi `Writer` hoàn thành -> Tự động chuyển qua `Critic` để audit, sau đó quay lại `Supervisor`.
   - Khi đã có `final_answer` (và hoàn tất audit) -> Điều hướng đến `END`.

## Guardrails

- **Max iterations**: Thiết lập cứng `max_iterations = 6` trong Supervisor để loại trừ hoàn toàn nguy cơ infinite loops.
- **Timeout**: Thiết lập `timeout_seconds = 60s` cho từng HTTP/LLM API call qua `Settings`.
- **Retry**: Áp dụng `tenacity` retry policy với exponential backoff (1s - 10s, tối đa 3 lần thử) khi gặp lỗi mạng/rate limit từ LLM provider.
- **Fallback**:
  - Search Client: Tự động fallback từ Tavily Live Search -> Local JSON Knowledge Corpus (`ai_agent_offline_research_corpus_v2`) -> Mock knowledge synthesis.
- **Validation**:
  - Kiểm tra tính hợp lệ của query đầu vào qua `ResearchQuery` (tối thiểu 5 ký tự, `max_sources` trong khoảng 1-20).
  - Pydantic BaseModel kiểm soát tính toàn vẹn kiểu dữ liệu của toàn bộ State và Agent Outputs.

## Benchmark plan

- **Tập truy vấn kiểm thử (Benchmark Test Queries)**:
  1. *"Research GraphRAG state-of-the-art and write a 500-word summary"* (Kỹ thuật chuyên sâu).
  2. *"Compare single-agent and multi-agent workflows for customer support"* (Bài toán đối sánh hệ thống).
  3. *"Summarize production guardrails for LLM agents"* (Thực hành kỹ thuật & an toàn hệ thống).
- **Chỉ số đo lường (Evaluation Metrics)**:
  - **Latency (Wall-clock seconds)**: Thời gian hoàn thành toàn bộ pipeline.
  - **Estimated Cost (USD)**: Tổng chi phí token suy luận (input/output) qua OpenAI model pricing.
  - **Quality Score (0 - 10)**: Điểm chất lượng nội dung đánh giá bởi Critic / Rubric (factuality, structure, completeness).
  - **Citation Coverage (0% - 100%)**: Tỷ lệ các luận điểm chính có trích dẫn nguồn tương ứng.
  - **Failure Rate (0% - 100%)**: Tỷ lệ request không sinh ra kết quả hợp lệ hoặc timeout.
- **Kỳ vọng kết quả (Hypothesis & Expected Outcome)**:
  - *Single-Agent Baseline*: Nhanh hơn (thấp hơn ~3x về latency), rẻ hơn (thấp hơn ~3x về cost), nhưng báo cáo khái quát hơn, dễ sót dẫn chứng.
  - *Multi-Agent Workflow*: Chất lượng chuyên sâu hơn, cấu trúc báo cáo chặt chẽ, trích dẫn chính xác và đầy đủ, chấp nhận đánh đổi (trade-off) về chi phí và thời gian chạy.

