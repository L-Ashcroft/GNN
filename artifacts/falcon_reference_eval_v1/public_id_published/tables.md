# Public ID: calibrated local readout

Historical fixed-test reanalysis; no model selection or new inference.

| Source | Samples | Full sample MRE (%) | Median (%) | P95 (%) |
| --- | ---: | ---: | ---: | ---: |
| AICircuit | 9926 | 12.5552 | 2.33543 | 40.7193 |
| FALCON | 19829 | 6.35973 | 2.28995 | 15.9315 |
| LOADBench | 19301 | 8.98374e+07 | 12.4518 | 1349.15 |

FALCON-only original-field pooling, following the reference table convention. This does not merge supervision.

| Metric | Display unit | Nonzero n | R² | RMSE | MAE | Full MRE (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bandwidth | GHz | 7926 | 0.996719 | 0.448024 | 0.213197 | 2.86963 |
| ConversionGain | dB | 3975 | 0.993195 | 0.992914 | 0.240877 | 5.15777 |
| DCPowerConsumption | mW | 17843 | 0.99964 | 0.535055 | 0.275701 | 2.21026 |
| DrainEfficiency | % | 3962 | 0.998182 | 0.619514 | 0.384054 | 5.60261 |
| NoiseFigure | dB | 7944 | 0.999382 | 0.145212 | 0.0611199 | 0.919086 |
| OscillationFrequency | GHz | 3966 | 0.941345 | 1.0912 | 0.463484 | 1.50221 |
| OutputPower | dBm | 3965 | 0.924052 | 1.23219 | 0.333181 | 5.92576 |
| PAE | % | 3962 | 0.997267 | 0.457275 | 0.270569 | 16.4742 |
| PSAT | dBm | 3962 | 0.990414 | 0.374244 | 0.222245 | 23.9756 |
| PhaseNoise | dBc/Hz | 3966 | 0.873793 | 2.95913 | 1.64253 | 1.85182 |
| PowerGain | dB | 7931 | 0.989752 | 0.425964 | 0.262501 | 20.5705 |
| S11 | dB | 7931 | 0.918333 | 1.60797 | 0.824188 | 9.91512 |
| S22 | dB | 3962 | 0.911841 | 1.50942 | 0.815983 | 10.0083 |
| TuningRange | GHz | 3966 | 0.771667 | 0.350263 | 0.136154 | 9.16418 |
| VoltageGain | dB | 3957 | 0.997311 | 0.31905 | 0.183733 | 7.73233 |
| VoltageSwing | mV | 3975 | 0.995185 | 0.423402 | 0.164253 | 2.84644 |

Absolute errors above use the explicitly listed display units; relative error uses restored native labels before display conversion. Physical diagnostics including observed zeros are retained in per_target_metrics.parquet.
Plot truncation is visualization only. Full summary metrics are untrimmed; the official-code tiny-trim variant is a separately named field.
