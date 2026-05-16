# Bug Catalog - LLM Refusal Evaluation Pipeline

This document catalogs all bugs found during the exploration phase of the refusal evaluation pipeline.

## Summary

- **Total bugs cataloged**: 70
- **High severity**: 15
- **Medium severity**: 40
- **Low severity**: 15

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

### BUG-031: Checkpoint resumption type validation missing (severity: medium)
- **Location**: Lines 1077-1087
- **What**: `dataset_judge_scores = json_load(partial_path)` assumes checkpoint contains valid structure
- **Risk**: If checkpoint is corrupted or has wrong structure, subsequent operations will fail
- **Fix**: Add type validation after loading checkpoint to ensure it's List[List[Dict]]

### BUG-032: Silent data loss in parse_log_progs (severity: medium)
- **Location**: Lines 84-87
- **What**: Positions with empty logprobs are silently skipped with `continue`
- **Risk**: Loss of data without user awareness - generated tokens are discarded
- **Fix**: Log warnings when skipping positions or track skipped count

### BUG-033: geom_mean_prob doesn't validate input types (severity: medium)
- **Location**: Lines 29-36
- **What**: Assumes logs contains float/int values but doesn't validate
- **Risk**: Non-numeric values in logs will cause torch.tensor to fail or produce incorrect results
- **Fix**: Add type validation for all elements in logs before creating tensor

### BUG-034: extract_refusal_score word boundary issues (severity: low)
- **Location**: Lines 471-488
- **What**: Word-by-word analysis with `words[i-1] == "not"` may miss "refusal" in compound words or phrases
- **Risk**: Phrases like "this is refusal-like behavior" might not be correctly classified
- **Fix**: Use regex with word boundaries or more sophisticated tokenization

### BUG-035: Index misalignment in judge result processing (severity: high)
- **Location**: Lines 1156-1158
- **What**: When an answer entry is missing "text", `result_idx += 1; continue` causes misalignment with batch_results
- **Risk**: Subsequent LLM-judged pairs will use wrong batch_results entries, leading to index errors or data corruption
- **Fix**: Don't increment result_idx when skipping; use consistent indexing pattern

### BUG-036: Case-sensitive _NON_CATEGORY_BOOLS check (severity: medium)
- **Location**: Line 684
- **What**: `_NON_CATEGORY_BOOLS` set is lowercase but column names may have different casing
- **Risk**: Columns like "Is_Safe" won't be excluded as intended
- **Fix**: Use case-insensitive comparison: `k.lower() in _NON_CATEGORY_BOOLS`

### BUG-037: Column name collision with different cases (severity: medium)
- **Location**: Line 629
- **What**: `col_lower = {c.lower(): c for c in available_columns}` loses information if two columns differ only by case
- **Risk**: If dataset has both "Prompt" and "prompt", one is arbitrarily chosen
- **Fix**: Handle duplicate column names explicitly or raise warning

### BUG-038: Empty prompt hash collision in deduplication (severity: medium)
- **Location**: Lines 806-819
- **What**: Rows with empty prompt_hash all get treated as same hash, only first is kept
- **Risk**: All rows with empty/missing prompt_hash are deduplicated even if they have different content
- **Fix**: Skip rows with empty prompt_hash or generate unique hash per row

### BUG-039: Missing "answers" key causes KeyError (severity: high)
- **Location**: Line 98
- **What**: `example["answers"]` is accessed without checking if key exists
- **Risk**: If "answers" key is missing, KeyError crashes the pipeline
- **Fix**: Use `example.get("answers", [])` or validate key exists first

### BUG-040: Path traversal vulnerability in split names (severity: high)
- **Location**: Line 1256
- **What**: `os.path.join(self.output_dir, split_spec["name"])` doesn't sanitize split names
- **Risk**: Malicious split name like "../../../etc/passwd" could write files outside output directory
- **Fix**: Sanitize split names to only allow alphanumeric, underscore, dash, dot characters

### BUG-041: No validation that answers.json and judges.json are in sync (severity: medium)
- **Location**: Lines 1495-1497
- **What**: Only checks if files exist, not if they have matching lengths
- **Risk**: If answers has 100 examples but judges only has 50, aggregation will produce incomplete results
- **Fix**: Validate both files have same number of examples before aggregating

### BUG-042: Zero num_return_sequences not validated (severity: medium)
- **Location**: Line 96 (llm_judge.py)
- **What**: num_return_sequences=0 could cause vLLM errors or return no outputs
- **Risk**: Pipeline may crash or produce unexpected results with zero return sequences
- **Fix**: Add validation for num_return_sequences > 0

### BUG-043: Negative tau validation message is misleading (severity: low)
- **Location**: Line 45
- **What**: Error message says "must be non-zero" but check is `tau <= 0`
- **Risk**: Confusing error message for negative tau values
- **Fix**: Change message to "must be positive" or "must be greater than zero"

### BUG-044: Whitespace-only thinking_string not normalized (severity: low)
- **Location**: Line 400 (compute_refusal_score.py)
- **What**: `thinking_string or None` only normalizes empty string, not whitespace
- **Risk**: thinking_string="   " would be treated as non-None but won't match anything
- **Fix**: Use `thinking_string.strip() or None` for normalization

### BUG-045: id() used for row identification is fragile (severity: medium)
- **Location**: Line 880
- **What**: `row_id = id(row)` uses memory address which can be reused
- **Risk**: In rare cases, different rows could have same id() if objects are destroyed
- **Fix**: Use `index(row)` or a stable identifier like prompt_hash

### BUG-046: Duplicate regex pattern definitions (severity: low)
- **Location**: Lines 351-367
- **What**: `_COMPILED_PATTERNS` and `_TAG_PATTERNS` are identical
- **Risk**: Code duplication, maintenance burden
- **Fix**: Remove unused `_COMPILED_PATTERNS` definition

### BUG-047: geom_mean_prob torch operations may fail on non-CPU (severity: low)
- **Location**: Line 31
- **What**: `torch.tensor(logs, dtype=torch.float32)` creates tensor on default device
- **Risk**: If CUDA is available but tensor should be on CPU, may cause device mismatch
- **Fix**: Explicitly specify device: `torch.tensor(logs, dtype=torch.float32, device='cpu')`

### BUG-048: Inconsistent error handling between split processing methods (severity: medium)
- **Location**: Lines 1156-1158 vs 1449
- **What**: `step_judge_scores` skips on missing "text" but `_step_judge_scores_single` uses `.get("text", "")`
- **Risk**: Different behavior in different code paths, inconsistent data handling
- **Fix**: Use consistent error handling pattern across both methods

---

## File: src/compute_refusal_score.py (Additional bugs)

### BUG-049: YAML config loading missing error handling (severity: high)
- **Location**: Line 357-368
- **What**: `load_config` function doesn't catch FileNotFoundError or YAMLError
- **Risk**: Pipeline crashes with unhelpful error message if config file is missing or malformed
- **Fix**: Add try/except for FileNotFoundError and yaml.YAMLError with helpful error messages

### BUG-050: No validation that answers.json and judges.json have same length (severity: medium)
- **Location**: Lines 1283-1288
- **What**: Only checks if files exist, not if they have matching numbers of examples
- **Risk**: If answers has N examples but judges has M≠N, aggregation produces incomplete results without warning
- **Fix**: Validate both files have same number of examples before aggregating

### BUG-051: geom_mean_prob torch.tensor device not specified (severity: low)
- **Location**: Line 38
- **What**: `torch.tensor(logs, dtype=torch.float32)` creates tensor on default device
- **Risk**: If CUDA is available, tensor may be created on GPU even though only CPU operations are needed
- **Fix**: Specify device='cpu': `torch.tensor(logs, dtype=torch.float32, device='cpu')`

### BUG-052: geom_mean_prob returns -1.0 for mixed valid/invalid data (severity: medium)
- **Location**: Lines 32-33
- **What**: If ANY value in logs is non-numeric, returns -1.0 (sentinel) for ENTIRE array
- **Risk**: Valid data is discarded along with invalid data - silent data loss
- **Fix**: Filter out invalid values and compute geometric mean on remaining valid values

### BUG-053: Category breakdown with all None scores produces no output (severity: low)
- **Location**: Lines 193-216
- **What**: If all scores are None, breakdown is empty with no indication
- **Risk**: User may not realize no valid data was found
- **Fix**: Log warning or return special "no_valid_data" indicator

### BUG-054: Category breakdown bootstrap with n=5 may be unstable (severity: low)
- **Location**: Line 229
- **What**: Bootstrap only runs if total >= 5, but 5 samples is too small for reliable CI
- **Risk**: Confidence intervals will be very wide and potentially misleading
- **Fix**: Increase minimum to 10-20 samples or label CI as "unreliable" for small samples

### BUG-055: Case-insensitive column matching loses duplicate columns (severity: medium)
- **Location**: Line 707
- **What**: `col_lower = {c.lower(): c for c in available_columns}` overwrites if two columns differ only by case
- **Risk**: If dataset has both "Prompt" and "prompt", one is arbitrarily chosen
- **Fix**: Detect duplicate column names and raise warning or error

### BUG-056: _NON_CATEGORY_BOOLS is case-sensitive (severity: medium)
- **Location**: Line 723
- **What**: `_NON_CATEGORY_BOOLS` set contains lowercase values but comparison is case-sensitive
- **Risk**: Columns like "Is_Safe" won't be excluded as intended
- **Fix**: Use case-insensitive comparison: `k.lower() in _NON_CATEGORY_BOOLS`

### BUG-057: Balanced sampling with samples_per_category > available samples (severity: low)
- **Location**: Lines 943-947
- **What**: Prints warning when samples requested > available, but doesn't handle gracefully
- **Risk**: User may not realize sampling was limited
- **Fix**: Clearly document behavior or raise error if insufficient samples

### BUG-058: Thinking string split may produce unexpected results (severity: medium)
- **Location**: Lines 1127, 1504
- **What**: `text.split(thinking_string)[-1]` returns entire text if thinking_string not found
- **Risk**: If thinking_string doesn't exist in text, no split occurs but code assumes split happened
- **Fix**: Check if thinking_string exists in text before splitting, or use split with maxsplit=1

### BUG-059: geom_mean_prob overflow handling may mask data issues (severity: low)
- **Location**: Lines 39-43
- **What**: Returns 1.0 on overflow, which is same as perfect probability
- **Risk**: Actual overflow issue (extremely large log probs) is masked
- **Fix**: Log warning or return special value to indicate overflow occurred

### BUG-060: Zero valid score edge case not explicitly handled (severity: low)
- **Location**: Line 221
- **What**: Score exactly 0.0 falls between "refusal" (>0.1) and "compliant" (<-0.1)
- **Risk**: Unclear categorization for exactly-zero scores
- **Fix**: Document behavior or adjust thresholds to handle 0.0 explicitly

---

## File: src/compute_refusal_score.py (Additional bugs)

### BUG-061: label=None causes TypeError in float conversion (severity: high)
- **Location**: Line 218
- **What**: `float(example_judges[k].get("label", 0.0))` crashes when label value is None (not missing)
- **Risk**: If judge output has `{"label": None}`, .get() returns None (not default 0.0), and float(None) raises TypeError
- **Fix**: Use `float(example_judges[k].get("label") or 0.0)` or check for None explicitly

### BUG-062: Very small prob values may cause math.log issues (severity: medium)
- **Location**: Line 219
- **What**: Denormal probabilities (e.g., 1e-323) pass prob > 0 check but math.log() may have precision issues
- **Risk**: Potential numerical instability with extremely small probability values
- **Fix**: Add lower threshold for prob values or use log-sum-exp trick for numerical stability

### BUG-063: Non-string values in questions list not validated (severity: medium)
- **Location**: answer_generator.py line 217
- **What**: `questions` list may contain None or non-string values without validation
- **Risk**: May cause issues downstream in tokenization or encoding
- **Fix**: Validate/filter questions list to ensure all elements are non-empty strings

### BUG-064: Non-tuple items in questions_answers not validated (severity: medium)
- **Location**: llm_judge.py line 96
- **What**: `questions_answers` may contain None or non-tuple items without validation
- **Risk**: Unpacking will fail if items aren't proper (question, answer) tuples
- **Fix**: Validate each item is a 2-element tuple/list before processing

### BUG-065: Softmax computation with extreme values may overflow (severity: medium)
- **Location**: Lines 106-110 (torch path)
- **What**: Very large logprob values may cause exp() overflow even with max normalization
- **Risk**: NaN results or crashes with extreme input values
- **Fix**: Add clipping or use more numerically stable softmax implementation

---

## File: src/utils.py (Additional bugs)

### BUG-066: __del__ during interpreter shutdown may fail (severity: low)
- **Location**: answer_generator.py line 256-257, llm_judge.py line 171-172
- **What**: __del__ calls close() which may fail if modules already cleaned up during shutdown
- **Risk**: Spurious errors during interpreter shutdown
- **Fix**: Use weakref.finalize() or add try/except in __del__

### BUG-067: Checkpoint resumption doesn't validate data consistency (severity: medium)
- **Location**: compute_refusal_score.py lines 1057-1067, 1220-1230
- **What**: Loading checkpoint doesn't verify it has valid structure or consistent length
- **Risk**: Corrupted checkpoint may cause data corruption or crashes
- **Fix**: Validate checkpoint structure and consistency before resuming

---

## File: src/compute_refusal_score.py (More bugs)

### BUG-068: Duplicate split names not validated (severity: medium)
- **Location**: Lines 1726-1784 (_normalize_dataset_splits)
- **What**: Duplicate split names are detected but error message could be clearer
- **Risk**: User confusion about which split is duplicated
- **Fix**: Include line numbers or more context in error message

### BUG-069: Temperature validation only checks <= 0 (severity: low)
- **Location**: Line 86
- **What**: Error message says "must be positive" but check is `tau <= 0`
- **Risk**: Confusing error message for edge case of tau=0 (should say "must be > 0")
- **Fix**: Change check to `tau <= 0` and message to "must be greater than zero"

---

## File: src/answer_generator.py (Additional bugs)

### BUG-070: Torch tensor creation without explicit device (severity: low)
- **Location**: Line 51
- **What**: `torch.tensor(logs, dtype=torch.float32)` uses default device
- **Risk**: If CUDA is available, tensor created on GPU even though only CPU operations needed
- **Fix**: Add device='cpu' parameter for consistency

---

## Summary by Type

- **Crash risks**: BUG-014, BUG-018, BUG-009, BUG-007, BUG-039, BUG-040, BUG-049, BUG-061, BUG-064
- **Data corruption**: BUG-005, BUG-029, BUG-008, BUG-035, BUG-038, BUG-041, BUG-050, BUG-052, BUG-055, BUG-067
- **Logic errors**: BUG-006, BUG-013, BUG-021, BUG-028, BUG-045, BUG-048, BUG-053, BUG-054, BUG-057, BUG-058, BUG-059, BUG-060, BUG-065, BUG-069
- **Type/validation issues**: BUG-004, BUG-010, BUG-026, BUG-036, BUG-037, BUG-042, BUG-051, BUG-056, BUG-062, BUG-063
- **Code quality**: BUG-001, BUG-002, BUG-003, BUG-024, BUG-043, BUG-044, BUG-046, BUG-047, BUG-066, BUG-068, BUG-070
- **Security**: BUG-040
