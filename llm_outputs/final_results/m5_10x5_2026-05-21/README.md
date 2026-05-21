# M5 LLM Final Results - 10x5 Slice

Generated on 2026-05-21.

This folder contains the final LLM recommendation outputs for the 10 anchor x 5 target/proposal slice. Each CSV has 50 rows. Each row contains 5 recommendations in one list field, plus the corresponding goodness scores.

## Layout

| Domain | Model | CSV | Raw responses |
|---|---|---|---|
| USC Teaming | Gemma 4 31B | `usc_teaming/gemma_4_31b/teaming_uc1_m5_gemma_4_31b_10prof_5prop.csv` | `usc_teaming/gemma_4_31b/raw_responses/` |
| USC Teaming | Gemini 3.1 Flash-Lite | `usc_teaming/gemini_3_1_flash_lite/teaming_uc1_m5_gemini_3_1_flash_lite_10prof_5prop.csv` | `usc_teaming/gemini_3_1_flash_lite/raw_responses/` |
| IITR Teaming | Gemini 3.1 Flash-Lite | `iitr_teaming/gemini_3_1_flash_lite/teaming_uc1_m5_gemini_3_1_flash_lite_10prof_5prop.csv` | `iitr_teaming/gemini_3_1_flash_lite/raw_responses/` |
| Meal | Gemini 3.1 Flash-Lite | `meal/gemini_3_1_flash_lite/meal_uc1_m5_gemini_3_1_flash_lite_10users_5items.csv` | `meal/gemini_3_1_flash_lite/raw_responses/` |

## Metrics

| Domain | Model | Rows | Anchors | Valid rows | G_mean | G_std | Volume_mean | Avg team/bundle size | Best | Min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| USC Teaming | Gemma 4 31B | 50 | 10 | 50 | 0.095304 | 0.007749 | 5.000000 | 2.828 | 0.156200 | 0.050000 |
| USC Teaming | Gemini 3.1 Flash-Lite | 50 | 10 | 50 | 0.082780 | 0.008714 | 5.000000 | 2.432 | 0.140600 | 0.050000 |
| IITR Teaming | Gemini 3.1 Flash-Lite | 50 | 10 | 50 | 0.056592 | 0.002481 | 5.000000 | 2.000 | 0.068800 | 0.050000 |
| Meal | Gemini 3.1 Flash-Lite | 50 | 10 | 50 | 0.843111 | 0.054162 | 5.000000 | 2.620 | 1.000000 | 0.750000 |

## Cleanup

Earlier smoke/sample M5 artifacts were removed from the domain output folders. The original non-LLM baseline outputs remain untouched.
