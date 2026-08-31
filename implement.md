# ViFinQA Implementation Plan

## 1. Mục tiêu và nguyên tắc

Nâng coverage và độ chính xác trên toàn bộ tập câu hỏi BTC, hiện gồm 1.012 ID. Không tạo output placeholder, không đổi NaN thành `0`, không dùng query hằng, không lưu evidence rỗng, không dùng evidence thiếu provenance. Complex question luôn fail-closed; không hạ xuống single-row fallback.

Thứ tự ưu tiên: acceptance/diagnostics → unit → retrieval/source mapping → planner → metric registry → extraction/complex solver → simple agent → semantic validation → GPU full run.

Không hard-code giả định ID liên tục trong gate. `expected_ids` phải đọc từ `questions.jsonl`; với corpus hiện tại tập này đúng `1..1012`.

## 2. Baseline đã xác minh

Nguồn baseline: `submission (2).zip`, `submission (1).json`, `submission.failures.json`, `submission.quality.json`, log và metrics BTC.

- ZIP SHA-256: `55b7272eae6ef515d49872895ecf85707a4efedb236a947ce0aaa4534e2fc2e4`.
- `submission.json` trong ZIP có SHA-256 `3d33cc4282899e966df7c4b245054fd26b96fc8e8783f0c0aafa857d0f917540`.
- 380 saved IDs, 632 failed IDs, không giao nhau, hợp đúng 1.012 expected IDs.
- Coverage saved: 37,55%. ZIP là validated partial package, không phải complete BTC submission.
- ZIP có 703 CSV; 703/703 được tham chiếu, không CSV thiếu hoặc thừa.
- 380/380 query replay trực tiếp từ ZIP thành công; 380/380 result khớp answer.
- Không answer/query/evidence rỗng, không query hằng, không biến `dfN` lệch evidence.
- Evidence fan-out: 281 câu dùng 1 CSV; có outlier dùng 40, 27, 20, 18 CSV.
- `TABLES_F2MACRO=0.075`, `DOCS_F2MACRO=0.2999`, `ANSWER_ACCURACY=0.1028`, `EXECUTION_ACCURACY=0.1028`. Replay đúng không chứng minh nghiệp vụ hoặc retrieval đúng.
- Failures: `missing_target_metric` 236, `missing_metric_facts` 113, query stage 105, semantic validation 77, `ambiguous_multiple_results` 37, `missing_metric_units` 14, `ambiguous_selection_metric` 13, `empty_filtered_subset` 6.
- Diagnostics chưa chuẩn: 428/632 thiếu `stage`, 204/632 thiếu `code`.
- Pytest collection chạy khi repo root có trong `PYTHONPATH`: 205 tests collected. Chưa coi đây là baseline pass cho đến khi chạy full suite.

## 3. Contract BTC bất biến

`submission.json` cuối là JSON array. Mỗi item có đúng bảy field, không thêm field nội bộ:

```python
BTC_FIELDS = {
    "id", "question", "answer", "relevant_docs",
    "relevant_tables", "evidence", "pandas_query",
}
```

Yêu cầu:

- `id`: integer duy nhất, thuộc `expected_ids`.
- `question`: string và khớp source question theo ID.
- `answer`: `int`/`float` hữu hạn; `0` chỉ hợp lệ khi query từ evidence thật tính ra `0`.
- `relevant_docs`: list document IDs có provenance.
- `relevant_tables`: list `<doc_id>|<line_number>`; line phải tồn tại và đúng source table, không chỉ đúng regex.
- `evidence`: list không rỗng; mỗi record chỉ có `variable`, `csv_path`; biến duy nhất; path dạng phẳng `data/<file>.csv`.
- `pandas_query`: một safe expression, tham chiếu chính xác tập biến evidence, replay ra answer.

ZIP cuối:

```text
submission.zip
├── submission.json
└── data/
    └── <mọi CSV và chỉ CSV được evidence tham chiếu>
```

ZIP phải replay độc lập, không dùng file ngoài ZIP. Chặn basename collision và duplicate archive entry.

## 4. Ba acceptance gate

### 4.1 `item_valid`

Một item chỉ được save khi:

- Đúng chính xác `BTC_FIELDS` sau strip.
- Answer numeric, finite.
- Evidence và provenance không rỗng.
- Query qua AST/restricted-expression validation.
- `referenced_variables(query) == evidence variables`.
- Query replay trên final evidence thành công và khớp answer.
- Semantic validation pass.
- Complex single-row fallback không được dùng.

### 4.2 `run_accounted`

Run hoàn chỉnh về accounting khi:

```text
saved_ids ∩ failure_ids = ∅
saved_ids ∪ failure_ids = expected_ids
```

Checkpoint có thể partial. Accounting pass không đồng nghĩa đủ điều kiện nộp.

### 4.3 `submission_ready`

Chỉ true khi:

```text
saved_ids = expected_ids
failure_ids = ∅
all items item_valid
ZIP-only replay pass
```

Không tạo hoặc ghi đè ZIP BTC nếu gate fail. Ghi checkpoint, failures và quality report để resume. Tạo ZIP atomically qua file tạm rồi rename sau khi tất cả checks pass.

## 5. Diagnostics và quality report

Mọi failure phải có schema thống nhất:

```json
{
  "id": 1,
  "question": "...",
  "stage": "retrieval",
  "code": "missing_metric_facts",
  "question_type": "multi_stage_analytical",
  "retryable": true,
  "retry_layer": "metric_retrieval",
  "target_metric": "...",
  "target_unit": "...",
  "requirements": [],
  "attempted_paths": [],
  "details": {},
  "message": "..."
}
```

Tách query failures thành machine-readable codes: syntax/forbidden construct, missing dataframe, missing column, empty subset, non-scalar, non-numeric, non-finite, evidence mismatch, answer mismatch.

Quality report tách ba nhóm:

- Safety: replay rate, answer/query consistency, finite answer, evidence closure, fallback counts.
- Coverage: saved/expected, failures theo stage/code/type, requirement coverage.
- Correctness: BTC answer accuracy, docs/tables F2, planner/fact/source audit accuracy.

Diagnostics nằm trong `.failures.json`, `.quality.json`, audit JSON và log; không lọt vào submission BTC.


## 6. Thiết kế semantic

### 6.1 Planner

`QuestionPlan` tách rõ:

- `target_phrase` và canonical `target_metric` nếu có.
- `subject_qualifier` và row hints cho open-vocabulary lookup.
- filter metric, selection metric, output metric.
- aggregation/grouping/domain.
- ticker, year, scope, statement type, period role.
- target unit và output kind.

Plan phải đầy đủ hoặc failure rõ. Không ép mọi cụm tài chính vào canonical metric gần nhất.

### 6.2 Metric registry

Phân loại 236 `missing_target_metric`; không thêm máy móc 236 aliases. Chỉ thêm canonical metric ổn định khi có:

- Semantic aliases không chồng lấn nguy hiểm.
- Required base metrics.
- Formula xác định và zero-denominator behavior.
- Output kind và unit contract.
- Regression tests tổng quát, không lookup theo question ID.

Statement rows mở dùng `target_phrase`, `subject_qualifier`, row hints thay vì làm registry phình sai.

### 6.3 Retrieval requirements

Retrieve theo từng tuple:

```text
(ticker, year, scope, metric_or_target_phrase,
 statement_type, period_role, subject_qualifier)
```

`EvidenceBundle` lưu ranked candidates, score components, rejection reasons và provenance theo requirement. Hard-filter entity/year/scope khi chắc chắn; preserve ít nhất một candidate cho từng requirement trước global ranking. Retry đúng requirement thiếu, không tăng `per_metric_k` đồng loạt.

Đo cả recall@1/@3/@5 và precision@1/@3/@5, requirement coverage, evidence fan-out, irrelevant evidence rate, exact doc/table/source-line accuracy. Evidence >10 cần diagnostic reason; không ép mọi câu về một bảng nếu semantics thật sự đa miền.

### 6.4 Extraction và solver

`SemanticExtractor` chọn fact theo metric/target phrase, qualifier, period, scope và unit; giữ source path, variable, row, value column, source/base unit. Ambiguity trả candidates và lý do, không tự chọn row yếu.

`ComplexSolver` giải domain, filter, selection, aggregation và output metric deterministically. Chỉ tạo query khi đủ facts có provenance. Giữ dấu số liệu; không dùng `abs()` hoặc zero fallback để che lỗi.

### 6.5 Unit

`detect_target_unit()` ưu tiên answer clause và output kind. Không fallback toàn câu theo cách bắt `cổ phần` từ `Công ty Cổ phần`. Phân biệt `%`, điểm phần trăm, lần, shares, VND/share, VND scales và USD; không invent exchange rate.

### 6.6 Simple agent

Truyền schema thật và final variable mapping. Structured retry cho missing column, empty subset, non-scalar và non-finite. Rule fallback chỉ dùng khi một row rõ ràng, period/scope/unit phù hợp và đủ provenance. Không mở fallback này cho complex question.

## 7. File changes dự kiến

- `src/pipeline.py`: gates, atomic packaging, complete failure schema, counters, targeted retry.
- `validate_submission.py`: schema/completeness/package/ZIP-only replay modes và non-zero exit code.
- `src/units.py`: target-clause detection và output-kind compatibility.
- `src/retriever.py`: requirement retrieval, ranking diagnostics, precision/recall, source mapping.
- `src/question_planner.py`: target/filter/selection/output semantics và open-vocabulary plan.
- `src/metric_registry.py`: chỉ thêm stable canonical definitions có contracts.
- `src/complex_solver.py`: fact extraction, provenance, period/scope/unit, deterministic solving.
- `src/agent.py`, `src/fallback.py`: schema-aware retry và conservative simple fallback.
- `src/semantic_validation.py`: dùng enriched plan/facts; chỉ chỉnh sau upstream fixes.
- Tests/audit scripts: đọc failures dạng list, audit đủ expected IDs, không giả định map.
- `environment.yml`/pytest config: đồng bộ dependency/import path sau khi xác nhận môi trường thật.

Không tạo planner/retriever song song mới; mở rộng contract hiện tại để tránh lệch hành vi.

## 8. Testing và acceptance theo phase

### Phase 0 — Environment và baseline

- Chuẩn hóa command test/import path.
- Chạy full suite lấy baseline pass/fail thật; 205 hiện mới là collection count.
- Khóa manifest cho 380 saved: question/query/answer/evidence/docs/tables hashes và replay status.
- Tạo regression buckets: top 20 mỗi nhóm lớn, toàn bộ nhóm nhỏ, saved outliers evidence 40/27/20/18.

### Phase 1 — Gates và diagnostics

- Validator báo riêng `SCHEMA PASS`, `COMPLETENESS PASS/FAIL`, `PACKAGE PASS/FAIL`.
- Baseline 380 phải schema/replay pass nhưng completeness fail thiếu 632 IDs.
- Partial run không tạo ZIP; full valid run mới tạo ZIP.
- 100% failures có `stage` và `code`.
- Accounting set invariant có tests cho duplicate, missing, extra và overlap IDs.

### Phase 2 — Unit

Regression cho `Công ty Cổ phần`, câu hỏi shares, `%`, điểm phần trăm, lần, VND/share, currency scales và incompatible dimensions. Giảm unit failures nhưng không nới semantic gate.

### Phase 3 — Retrieval/source mapping

Đo recall và precision docs/tables trước solver. Test requirement coverage, stable ranking, source line tồn tại/đúng table và minimal final evidence. Audit evidence fan-out outliers.

### Phase 4 — Planner và registry

Test target/filter/selection/output split, multi-company/year, scope, period role, aggregation và open-vocabulary rows. Đo plan completeness trên 1.012. `missing_target_metric` giảm mà ambiguous/wrong canonical mapping không tăng.

### Phase 5 — Extraction và complex solver

Test exact rows, period/scope, units, formulas, signs, zero denominators, variables và replay. `missing_metric_facts`/`missing_metric_units` giảm; complex fallback vẫn bằng 0.

### Phase 6 — Simple agent

Regression cho 105 query-stage cases, nhất là 104 non-finite symptoms. Không query absent column, không `.iloc[0]` trên empty subset, không NaN-to-zero.

### Phase 7 — Validation tuning

Chỉ giảm false positive sau khi facts đúng. Safety invariants, answer/query consistency và complex fail-closed không giảm.

### Phase 8 — Full run

- Chạy targeted CPU tests, full suite, representative GPU buckets.
- Chạy full expected IDs từ checkpoint sạch.
- So saved/failures và BTC correctness metrics với baseline.
- Chỉ package khi `submission_ready`.
- Replay toàn bộ query từ chính ZIP trước nộp.

## 9. Thứ tự triển khai

1. Khóa checksums và baseline manifests cho ZIP/JSON/failures/quality/log.
2. Sửa test import environment; chạy full suite lấy baseline thật.
3. Tạo accounting fixtures cho 380 saved + 632 failed = expected IDs.
4. Thêm `item_valid`, `run_accounted`, `submission_ready` và validator modes.
5. Chặn ZIP incomplete; package atomically; thêm ZIP-only replay.
6. Chuẩn hóa failure schema, bắt buộc `stage` và `code`.
7. Sửa unit false positives và output-kind compatibility.
8. Audit fan-out; sửa retrieval ranking, requirement coverage và source mapping.
9. Sửa planner target/filter/selection/output/domain/scope/period semantics.
10. Phân loại `missing_target_metric`; mở rộng registry có kiểm soát.
11. Sửa fact extraction và unit/period/scope provenance.
12. Sửa complex solver; giữ complex fallback bằng 0.
13. Sửa simple agent empty-subset/schema/NaN path và conservative fallback.
14. Hiệu chỉnh semantic validation sau upstream fixes.
15. Chạy targeted tests, full CPU suite và representative GPU buckets.
16. Chạy full 1.012 từ checkpoint sạch; đo metrics và replay ZIP.
17. Chỉ package/nộp khi mọi gate pass và correctness tốt hơn baseline.

## 10. Definition of done

Bắt buộc:

- Submission cuối có đúng expected IDs, duy nhất và đúng question mapping.
- Mỗi item có đúng 7 fields BTC.
- 100% answer finite, evidence có provenance, safe query replay được và khớp answer.
- ZIP có đúng CSV closure, không missing/thừa/collision.
- Không placeholder, constant query, empty evidence, NaN-to-zero.
- Complex single-row fallback bằng 0.
- Failure/accounting diagnostics đầy đủ cho mọi run chưa hoàn tất.

Cải thiện phải đo được:

- Saved coverage tăng mà audited correctness không giảm.
- `ANSWER_ACCURACY`, `EXECUTION_ACCURACY`, `DOCS_F2MACRO`, `TABLES_F2MACRO` tăng so baseline.
- `missing_target_metric`, `missing_metric_facts`, query non-finite và source mapping errors giảm.
- Retrieval precision/recall và requirement coverage tăng; evidence fan-out không cần thiết giảm.

Mục tiêu kỹ thuật là 1.012/1.012 valid outputs. Nếu còn failure, giữ checkpoint và diagnostics, không lấp bằng output đoán và không tạo ZIP BTC giả hoàn chỉnh.

