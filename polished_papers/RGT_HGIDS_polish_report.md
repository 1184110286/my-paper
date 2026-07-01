# RGT-HGIDS polishing report

## Scope

Polished the uploaded `RGT_HGIDS_SCI_Methods_with_Figures.docx` methods section using the supplied project code/documentation and the writing skills from `1184110286/my-paper`. The final DOCX preserves the original figures, tables, equations-as-text, captions, and section order.

## Skill routing used

- `ppw:polish`, direct mode: academic English polishing with terminology preservation, simple and clear vocabulary, no added emphasis, and anti-AI phrase screening.
- `nature-polishing`, detected axes: `paper_type=algorithmic`, `section=methods`, `language=en`, `journal=generic`.
- DOCX render-and-verify workflow: the edited DOCX was rendered to page PNGs and inspected page by page.

## Terminology ledger

| Canonical term | First-use form | Decision |
|---|---|---|
| RGT-HGIDS | Redundancy-aware Gated Temporal Heterogeneous Graph Intrusion Detection System (RGT-HGIDS) | Spell out once, then use RGT-HGIDS. |
| TBB-RR | Target-Budget Boundary Redundancy Reduction (TBB-RR) | Keep as the canonical redundancy module name. |
| RGD-BiGRU-MCBG | Residual Gated Dilated BiGRU Multi-Channel Behavior Graph encoding (RGD-BiGRU-MCBG) | Keep as the canonical semantic encoder name. |
| ST-HGAN | Spatial-Temporal Heterogeneous Graph Attention Network propagation (ST-HGAN) | Keep the manuscript's spelling and acronym. |
| EHA | Elastic Hop Aggregation (EHA) | Keep as the canonical hop-fusion module. |
| MCBG | Multi-Channel Behavior Graph | Use MCBG for the baseline branch after first mention. |
| AP, MCC, ROC-AUC | average precision, Matthews correlation coefficient, ROC-AUC | Define AP and MCC where metrics are introduced. |

## Major edits

- Reduced overloaded sentences and separated system definition, mechanism rationale, and experimental protocol.
- Strengthened methods-section reproducibility language by keeping parameters, seeds, thresholds, and ablation conditions explicit.
- Removed generic or inflated phrasing where possible, especially around robustness, improvement, and contribution claims.
- Standardized terminology across the pipeline and tables.
- Preserved all formulas and did not invent data, results, references, or new claims.

## Visual QA

Rendered final file: `RGT_HGIDS_SCI_Methods_polished.docx` -> 7 pages. Page-by-page PNG inspection found no clipping, overlap, broken tables, missing figures, or header/footer displacement.

## Deliverable note

The polished DOCX and PDF were generated in this ChatGPT workspace. Because the standard GitHub content action available here accepts UTF-8 text, this repository record contains the polishing report; the DOCX/PDF deliverables are returned in the chat as downloadable files.