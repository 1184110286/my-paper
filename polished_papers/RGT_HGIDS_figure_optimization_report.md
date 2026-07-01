# RGT-HGIDS figure optimization report

## Scope

Optimized the four manuscript figures in the polished RGT-HGIDS Methods document. The DOCX was updated by replacing the embedded raster figures while preserving the manuscript text, captions, tables, pagination style, and section order.

## Skill routing used

- `nature-figure`: Python backend, schematic-led composite / methods-figure workflow.
- `ppw:caption` and `ppw:visualization` principles were applied as constraints: each figure must support a specific claim, avoid redundant panels, keep labels concise, and use a consistent visual vocabulary.
- DOCX render-and-verify workflow: the updated manuscript was rendered to PDF and page PNGs for visual QA.

## Figure contract

| Figure | Core conclusion | Main visual change |
|---|---|---|
| Figure 1 | RGT-HGIDS is a fixed staged pipeline, and the paired evaluation isolates the semantic encoder comparison. | Rebuilt as a pipeline plus paired-evaluation contract strip with consistent colors and direct labels. |
| Figure 2 | TBB-RR retains block boundaries under a target budget while preserving event order. | Rebuilt event blocks, retained/skipped encoding, compressed sequence, and operation chain. |
| Figure 3 | RGD-BiGRU-MCBG combines gated local filtering, recurrent context, self-attention, and weighted pooling. | Rebuilt as a top-level pipeline plus an internal RGD-block subflow. |
| Figure 4 | ST-HGAN propagates relation/time-aware graph information, and EHA fuses hop outputs before classification. | Rebuilt as a schematic-led composite: provenance graph, ST-HGAN, hop outputs, EHA, and classifier. |

## Design choices

- Used a restrained white-background palette with stable color semantics across panels: blue for sequence/graph representation, teal for compression, green for RGD-BiGRU-MCBG, orange for ST-HGAN, purple for EHA, and rose for classification.
- Reduced long text inside boxes and moved explanatory details to captions or concise note strips.
- Used direct labels and shared visual vocabulary rather than repeated legends.
- Exported high-resolution PNGs for the DOCX and retained SVG/PDF/TIFF source exports in the working package.
- Added alt-text descriptions to the four embedded figures.

## QA

- Final DOCX rendered successfully to a 7-page PDF.
- Page PNG inspection found no missing figures, overlap, clipping, broken captions, or table displacement.
- Figure dimensions were kept compatible with the existing Word layout.

## Deliverable note

The updated DOCX, PDF preview, individual figure exports, and a zipped figure package were generated in the ChatGPT workspace. This GitHub record documents the optimization because the available GitHub text action is best suited to UTF-8 repository files rather than binary DOCX/PDF uploads.