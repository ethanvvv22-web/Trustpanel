Held-out products: 60. Values with 95% bootstrap intervals.

| Metric | Persona panel | Persona panel, calibrated | Generic LLM | Generic LLM, calibrated | Topic prior | Better |
|---|---|---|---|---|---|---|
| Mean rating error (stars) | 0.82 [0.66, 1.00] | 0.36 [0.30, 0.42] | 0.41 [0.36, 0.47] | 0.32 [0.26, 0.40] | 0.33 [0.28, 0.40] | lower |
| Distribution distance (stars) | 1.19 [1.07, 1.33] | 0.37 [0.31, 0.43] | 1.10 [1.04, 1.16] | 0.34 [0.28, 0.40] | 0.34 [0.29, 0.40] | lower |
| Error in share of 1-2 star reviews | 0.22 [0.17, 0.27] | 0.09 [0.07, 0.10] | 0.15 [0.12, 0.17] | 0.08 [0.06, 0.10] | 0.08 [0.07, 0.10] | lower |
| Picks the better product (pairs) | 0.58 [0.39, 0.77] | 0.42 [0.23, 0.61] | 0.56 [0.45, 0.65] | 0.56 [0.45, 0.65] | 0.50 [0.50, 0.50] | higher |
| Rank correlation with real means | 0.17 [-0.10, 0.42] | -0.17 [-0.43, 0.10] | 0.18 [-0.08, 0.39] | 0.03 [-0.23, 0.29] | 0.22 [-0.00, 0.42] | higher |
| Bias (predicted - real mean) | -0.71 [-0.93, -0.53] | -0.06 [-0.18, 0.05] | -0.21 [-0.32, -0.10] | -0.05 [-0.16, 0.05] | -0.05 [-0.16, 0.05] | closer to 0 |
| Spread vs real spread | 0.45 [0.39, 0.52] | 1.13 [1.07, 1.22] | 0.04 [0.02, 0.07] | 1.12 [1.06, 1.21] | 1.12 [1.06, 1.20] | closer to 1 |

Mean rating error by topic:

| Topic | Persona panel | Persona panel, calibrated | Generic LLM | Generic LLM, calibrated | Topic prior |
|---|---|---|---|---|---|
| baby_care | 1.31 | 0.21 | 0.33 | 0.23 | 0.20 |
| bath_tissue | 0.78 | 0.32 | 0.36 | 0.23 | 0.30 |
| feminine_care | 0.54 | 0.36 | 0.41 | 0.36 | 0.38 |
| kitchen_paper | 0.77 | 0.45 | 0.49 | 0.45 | 0.46 |
| skin_care | 1.10 | 0.34 | 0.47 | 0.31 | 0.23 |
| wound_care | 0.46 | 0.46 | 0.44 | 0.35 | 0.44 |
