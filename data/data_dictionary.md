# Data Dictionary

Input files are expected to be in CSV format with the following schema:

| Column Name | Data Type | Description | Range / Constraints |
|---|---|---|---|
| `age` | Float | The chronological age of the sample | $> 0$ |
| `cg[0-9]{8}` | Float | Methylation beta value for the specified CpG site | $[0, 1]$ |

**Missing Values:** Missing beta values are imputed to `0.5` by the system (representing maximum informational entropy).
