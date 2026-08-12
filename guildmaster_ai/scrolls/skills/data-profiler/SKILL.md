---
name: data-profiler
description: Profile CSV datasets to get summary statistics — row counts, column types, missing values, and numeric distributions. Use when asked to analyze, describe, or profile tabular data.
---

When asked to profile or analyze a CSV dataset:

1. Run the profiling script on the target file:
   ```
   run_script(path="<skill_dir>/scripts/profile_csv.py", args="<path_to_csv>")
   ```
   Replace `<skill_dir>` with the scroll's resource directory path and `<path_to_csv>` with the actual file path.

2. The script outputs a JSON report with:
   - `row_count` and `column_count`
   - Per-column: `dtype`, `missing`, `unique`, and for numeric columns: `mean`, `min`, `max`

3. Summarize the results for the user in a clear, readable format.
   Highlight any data quality issues (high missing rates, unexpected types, etc.).
