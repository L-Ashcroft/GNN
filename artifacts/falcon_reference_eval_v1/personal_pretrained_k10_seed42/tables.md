# Personal e2e: pretrained, k=10, seed=42

Historical fixed-test reanalysis; no model selection or new inference.

| Source | Samples | Full sample MRE (%) | Median (%) | P95 (%) |
| --- | ---: | ---: | ---: | ---: |
| AnalogGym_generated | 2000 | 213.587 | 56.3586 | 309.301 |

Absolute errors above use the explicitly listed display units; relative error uses restored native labels before display conversion. Physical diagnostics including observed zeros are retained in per_target_metrics.parquet.
Plot truncation is visualization only. Full summary metrics are untrimmed; the official-code tiny-trim variant is a separately named field.
