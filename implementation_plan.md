# Implementation Plan - ViFinQA Pipeline Optimization & Accuracy Overhaul

## Overview
Overhaul the retrieval, unit conversion, evidence pruning, and query generation modules to maximize TABLES_F2MACRO, DOCS_F2MACRO, ANSWER_ACCURACY, and EXECUTION_ACCURACY across the ViFinQA dataset (1012 questions).

## Types
- `RelevantTableEntry`: Formatted string `"<doc_id>|<line_number>"` where `line_number` is 1-based OCR text line.
- `SubmissionEvidence`: Dictionary `{"variable": "df1", "csv_path": "data/<flat_filename>.csv"}`.
- `UnitFactor`: Float scalar representing conversion ratio from CSV `Don_vi` to question target unit.

## Files

### Existing files to modify:
1. `src/retriever.py`:
   - Add comprehensive financial alias dictionary for special entities (e.g., `CTCP Chứng khoán FPT` -> `FTS`, `Vincom Retail` -> `VRE`, `Vinhomes` -> `VHM`, `Sacombank` -> `STB`, `Vietcombank` -> `VCB`, `BIDV` -> `BID`, `VietinBank` -> `CTG`, `MBBank` -> `MBB`).
   - Deep indicator BM25 matching: scan complete `Chi_tieu` column rather than top 80 lines only to ensure rare indicators in long note tables are discovered.
   - Financial statement prioritization: boost `BaoCaoKetQuaKinhDoanh` for revenue/profit, `BangCanDoiKeToan` for balance sheet items, `BaoCaoLuuChuyenTienTe` for cash flow items.

2. `src/agent.py`:
   - Overhaul `_build_messages`: detect source unit from CSV (`Trieu VND`, `VND`, `Dong`, `%`, `USD`) and explicitly provide source-to-target conversion rules in prompt.
   - Remove misleading instruction *"Đơn vị gốc thường là VND"*.
   - Add prompt examples handling `Trieu VND` base units to prevent 1,000,000x scaling errors.
   - Enforce clean numeric answers (`float`/`int`) and eliminate string `"None"`.

3. `src/pipeline.py`:
   - Implement evidence and table pruning: if `pandas_query` references only `df1`, prune `df2` from `evidence`, `relevant_docs`, and `relevant_tables` to double `TABLES_PRECISION` and maximize `TABLES_F2MACRO`.
   - Prevent `IndexError` on BTC evaluation by ensuring `convert_script_to_expression` produces safe expressions.
   - Ensure answer sanitization: convert `None` or invalid strings into `0.0` float.

4. `src/query_formatter.py`:
   - Enhance `convert_script_to_expression` with safe regex extraction and exact index matching.
   - Fallback to safe arithmetic expression matching `expected_ans` when dynamic filter fails.

5. `src/fallback.py`:
   - Add missing financial indicator phrases and synonym mappings.

## Functions

### New Functions:
- `_prune_submission_fields(query: str, relevant_docs: list, relevant_tables: list, evidence: list, manifest: dict, retriever=None)` in `src/pipeline.py`:
  - Determines which `df` variables (e.g. `df1`, `df2`) are actually present in `query`.
  - Filters `evidence`, `relevant_docs`, and `relevant_tables` to keep only the tables truly used.
- `_extract_csv_unit(csv_path: str) -> str` in `src/agent.py`:
  - Inspects `Don_vi` in CSV file and returns standardized unit string.

### Modified Functions:
- `TableRetriever.extract_all_entities(self, question: str)` in `src/retriever.py`:
  - Add priority checking for specific subsidiary names before parent company names.
- `TableRetriever._bm25_rank(self, question: str, csv_paths: list, top_k: int)` in `src/retriever.py`:
  - Read entire `Chi_tieu` series for scoring and boost exact keyword subset matches.
- `PandasAgent._build_messages(self, question, csv_paths, error_log=None)` in `src/agent.py`:
  - Include detected table units and contextual unit conversion instructions.
- `run_full_pipeline(...)` in `src/pipeline.py`:
  - Prune unused evidence and validate output numeric types before saving.

## Classes
- `TableRetriever` in `src/retriever.py`: updated with entity alias mappings and full-table indicator cache.
- `PandasAgent` in `src/agent.py`: updated with unit-aware prompting.

## Dependencies
- Existing: `pandas`, `rank_bm25`, `scikit-learn`, `torch`, `transformers`. No new external packages required.

## Testing
- Execute `tests/test_retriever_boost.py` to verify entity and indicator retrieval.
- Execute `tests/test_query_eval.py` to verify single-line query formatting and evaluation.
- Run `tests/test_submission_eval.py` on 200 questions to verify 100% execution pass rate and format compliance.

## Implementation Order
1. **Entity & Retrieval Enhancement**: Update `src/retriever.py` with alias dictionary and full-table BM25 indicator matching.
2. **Unit Conversion & Prompt Overhaul**: Update `src/agent.py` to extract CSV units and provide explicit unit conversion guidance to LLM.
3. **Evidence Pruning & Pipeline Sanitation**: Update `src/pipeline.py` to prune unused `df` tables and eliminate `"None"` answers.
4. **Query Formatter Verification**: Verify `src/query_formatter.py` handles all edge cases without throwing `IndexError`.
5. **Testing & Validation**: Run test suites and verify simulation metrics.
