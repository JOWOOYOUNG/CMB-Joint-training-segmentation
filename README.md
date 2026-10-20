# CMB-Joint-training-segmentation
Enhancing Cerebral Microbleeds Detection via Jointly Incorporating Multiple MR Sequences Validated in Multicenter SWI-only Inference
TL;DR. We train cerebral microbleed (CMB) segmentation models with SWI, filtered phase and QSM, but deploy them with SWI alone. Standard channel-wise multi-sequence nnU-Nets collapse when only SWI is available at test time (Dice 0.448 / 0.367 on two external cohorts). Joint training avoids this collapse (Dice 0.630 / 0.536, both p < 0.001) and matches or modestly exceeds an SWI-only baseline, including on a 48-institution external cohort.

<Why this matters>
- CMB counts now gate anti-amyloid therapy. Appropriate use recommendations for lecanemab exclude patients with more than four CMBs, so counting must be accurate and reproducible.
- QSM helps, but is rarely acquired. QSM separates hemorrhage from veins and calcification better than SWI, yet most hospitals only have SWI.
- Multi-sequence models usually assume every input at test time. Their behavior when a sequence is missing is rarely reported. We test it directly.

