# Bug Catalog - LLM Refusal Evaluation Pipeline

This document catalogs all bugs found during the exploration phase of the refusal evaluation pipeline.

## Summary

- **Total bugs cataloged**: 30
- **High severity**: 8
- **Medium severity**: 17
- **Low severity**: 5

---

## File: src/compute_refusal_score.py

### BUG-001: Duplicate method definition (severity: low)
- **Location**: Lines 468-470
- **What**: The `_get_answer_generator` method is defined twice (lines 468-470 duplicate lines 469-482)
- **Risk**: The second definition shadows the first, but in this case they are identical so no functional impact. Code maintenance issue.
- **Fix**: Remove the duplicate definition at lines 468-470

### BUG-002: Duplicate empty check (severity: low)
- **Location**: Lines 290-293
- **What**: `if len(valid_values) == 0:` is checked twice consecutively
- **Risk**: Code duplication, minor maintenance issue
- **Fix**: Remove one of the duplicate checks

### BUG-003: Redundant prob check (severity: low)
- **Location**: Lines 137-142
- **What**: After checking `prob is None or prob <= 0` at line 137-142, the same condition is checked again at line 137
- **Risk**: Code duplication, but no functional impact due to `continue` statement
- **Fix**: Remove the redundant check at lines 137-142

### BUG-004: Inconsistent NaN comparison (severity: medium)
- **Location**: Line 289
- **What**: NaN check uses `v != v` which is Python-specific behavior instead of `math.isnan(v)`
- **Risk**: While `v != v` works for NaN in IEEE 754, it's less readable and may not work consistently across all numeric types
- **Fix**: Use `math.isnan(v)` for clarity

### BUG-005: No validation of avg_logprobs/labels length mismatch (severity: high)
- **Location**: Lines 152-159
- **What**: `aggregate_with_softmax` is called with `avg_logs` and `labels` but no validation that they have the same length
- **Risk**: If lengths differ, torch operations may silently broadcast or produce incorrect results
- **Fix**: Add length validation before calling aggregate_with_softmax

### BUG-006: Missing validation for tau parameter (severity: medium)
- **Location**: Line 43-44
- **What**: Only checks `tau == 0` but doesn't check for negative values
- **Risk**: Negative tau could produce unexpected softmax behavior
- **Fix**: Check `tau <= 0`

### BUG-007: math.log on prob <= 0 passes then crashes (severity: high)
- **Location**: Lines 129, 143
- **What**: Code warns on `prob <= 0` but continues, then `math.log(prob)` raises ValueError for prob=0
- **Risk**: Runtime crash when prob is exactly 0.0
- **Fix**: The continue statement at line 132 should be executed for prob <= 0, not just prob < 0

### BUG-008: Best answer selection uses 0.0 default for missing answer_prob (severity: medium)
- **Location**: Line 167
- **What**: `max(example_answers, key=lambda a: a.get("answer_prob", 0) or 0)` may select answers with missing/invalid answer_prob
- **Risk**: If all answers have invalid answer_prob, an arbitrary answer is selected
- **Fix**: Filter out invalid answers before finding max, or use explicit sentinel value

### BUG-009: No GPU availability check before torch.cuda.device_count() (severity: high)
- **Location**: Line 396
- **What**: `torch.cuda.device_count()` is called without checking if CUDA is available
- **Risk**: Will crash on systems without CUDA
- **Fix**: Check `torch.cuda.is_available()` first

### BUG-010: Division by zero potential in compliance percentage (severity: medium)
- **Location**: Lines 300-309
- **What**: `total_count = float(len(valid_values))` but no explicit check if it's 0 before division
- **Risk**: If valid_values is empty, division by zero will occur
- **Fix**: The code does have `if total_count > 0` checks, but they could be more explicit

### BUG-011: Missing keys in example dict not handled (severity: medium)
- **Location**: Lines 87-96
- **What**: Uses `.get()` for optional keys but doesn't validate required keys exist
- **Risk**: Missing required keys will cause None values in output
- **Fix**: Validate required keys exist before processing

### BUG-012: Potential issue with large bootstrap sampling (severity: low)
- **Location**: Line 203
- **What**: `np.random.default_rng(42)` is created per category
- **Risk**: Minor performance issue with large datasets
- **Fix**: Consider caching the rng instance

---

## File: src/answer_generator.py

### BUG-013: geom_mean_prob doesn't handle all edge cases (severity: medium)
- **Location**: Lines 20-37
- **What**: Returns -1.0 for empty logs, but doesn't handle case where logs contain NaN or inf
- **Risk**: NaN/inf in logs will propagate through exp() and cause incorrect results
- **Fix**: Add NaN/inf validation before computing avg_log

### BUG-014: ValueError in parse_log_progs when values is empty (severity: high)
- **Location**: Lines 84-89
- **What**: `chosen = max(values, key=lambda x: x.rank)` will raise ValueError if values is empty
- **Risk**: Runtime crash when logprobs sequence contains empty mappings
- **Fix**: Add explicit check for empty values before calling max()

### BUG-015: No validation of negative temperature (severity: medium)
- **Location**: N/A (temperature not validated in SamplingParams)
- **What**: Temperature parameter is not validated for negative values
- **Risk**: Negative temperature produces undefined behavior in sampling
- **Fix**: Add validation for temperature >= 0

### BUG-016: Missing validation of max_new_tokens (severity: medium)
- **Location**: Line 145
- **What**: `max_new_tokens` is not validated to be positive
- **Risk**: Zero or negative max_new_tokens may cause issues
- **Fix**: Add validation for max_new_tokens > 0

### BUG-017: Empty questions list returns [] but caller may expect error (severity: low)
- **Location**: Lines 169-170
- **What**: Returns empty list for empty questions without indication of no-op
- **Risk**: Caller may not distinguish between "no input" and "empty result"
- **Fix**: Consider logging or raising error for empty input

---

## File: src/llm_judge.py

### BUG-018: tensor_parallel_size default crashes on non-CUDA systems (severity: high)
- **Location**: Line 35
- **What**: `tensor_parallel_size: int = torch.cuda.device_count()` as default parameter
- **Risk**: Will crash on systems without CUDA when CUDA is not available
- **Fix**: Use None as default and handle in __init__

### BUG-019: Inconsistent empty string handling for thinking_string (severity: low)
- **Location**: Lines 111-112
- **What**: Empty string is normalized to None in judge() but not consistently across the codebase
- **Risk**: Inconsistent behavior if thinking_string="" is set
- **Fix**: Normalize in __init__ for consistency

### BUG-020: No validation of judge output structure (severity: medium)
- **Location**: Lines 121-146
- **What**: Assumes output structure has `.outputs` and each output has `.text`
- **Risk**: Will crash if vLLM output structure changes
- **Fix**: Add defensive checks for output structure

---

## File: src/utils.py

### BUG-021: ValueError check after encoding wastes resources (severity: medium)
- **Location**: Lines 117-120
- **What**: Checks `max_model_len <= max_new_tokens` AFTER encoding the conversation
- **Risk**: Wastes CPU encoding invalid conversations
- **Fix**: Move check before encoding

### BUG-022: delete_llm complex try/except may not cover all vllm versions (severity: medium)
- **Location**: Lines 41-50
- **What**: Multiple shutdown attempts with try/except, but may miss future vllm versions
- **Risk**: Incomplete cleanup or exceptions in future versions
- **Fix**: Consider version detection or more robust cleanup

### BUG-023: No check for max_model_len being None or zero (severity: medium)
- **Location**: Line 117
- **What**: No validation that max_model_len is positive
- **Risk**: Division or comparison errors if max_model_len is invalid
- **Fix**: Add validation for max_model_len > 0

---

## File: src/refusal_judge_prompt.py

### BUG-024: Duplicate comment line (severity: low)
- **Location**: Lines 469-470
- **What**: Comment "# Last resort: check for exact "not refusal" anywhere in text" appears twice
- **Risk**: Code duplication issue
- **Fix**: Remove duplicate comment

### BUG-025: Word-by-word analysis may miss multi-word patterns (severity: low)
- **Location**: Lines 476-488
- **What**: Splits by whitespace and checks individual words, may miss "refusal" in hyphenated or contrived formats
- **Risk**: May fail to classify valid refusals
- **Fix**: Use regex or more sophisticated tokenization

---

## File: src/compliance_quality.py

### BUG-026: Type validation insufficient (severity: medium)
- **Location**: Line 38-39
- **What**: Checks `if not response or not isinstance(response, str)` but `not response` is True for empty string (valid case)
- **Risk**: Returns 0.0 for empty string which may not be desired
- **Fix**: Separate None check from empty string check

### BUG-027: Hedge phrase matching is case-sensitive but response_lower is used (severity: low)
- **Location**: Line 52
- **What**: Code is correct (uses response_lower), but HEDGE_PHRASES are all lowercase - no actual bug
- **Risk**: None, but could be confusing
- **Fix**: Document or add assertion

---

## File: src/dataset_adapters.py

### BUG-028: Prefix matching could match unintended datasets (severity: medium)
- **Location**: Line 40
- **What**: `dataset_id_lower.startswith(prefix.lower())` could match "my-PKU-dataset" to "PKU-Alignment/BeaverTails"
- **Risk**: Wrong adapter applied to datasets with similar names
- **Fix**: Use more specific matching or full dataset ID matching

---

## File: merge_results.py

### BUG-029: Empty prompt hash creates collision risk (severity: medium)
- **Location**: Lines 34-36
- **What**: Empty string hash is computed for missing prompts, all missing prompts get same hash
- **Risk**: All missing/None prompts will be treated as duplicates
- **Fix**: Use a unique sentinel or skip entries with missing prompts

### BUG-030: No validation of merged data structure (severity: low)
- **Location**: Lines 189-196
- **What**: Writes merged data without validating structure consistency
- **Risk**: May produce corrupted output if input files have inconsistent structures
- **Fix**: Add validation or schema check

---

## Summary by Type

- **Crash risks**: BUG-014, BUG-018, BUG-009, BUG-007
- **Data corruption**: BUG-005, BUG-029, BUG-008
- **Logic errors**: BUG-006, BUG-013, BUG-021, BUG-028
- **Type/validation issues**: BUG-004, BUG-010, BUG-026
- **Code quality**: BUG-001, BUG-002, BUG-003, BUG-024
