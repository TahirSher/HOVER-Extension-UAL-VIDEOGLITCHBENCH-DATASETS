# CRUX v10 on UAL-Bench — implementation notes

`crux_v10.py` is one script. Every VideoGlitchBench command (`selftest`, `prepare`,
`diagnose`, `probe`, `calibrate`, `run`, `import_glide`, `export`, `evaluate`,
`ablate`) behaves as before. UAL-Bench is a separate, explicitly selected path:
`inspect_ual`, `run_ual`, `evaluate_ual`, `selftest_ual`.

## What changed in `crux_v10.py`

| Where | Change | Why |
|---|---|---|
| module docstring | UAL-Bench section added | usage |
| new block before `# CLI` | UAL loader, Mode A, official evaluator port, statistics, commands, `selftest_ual` | the whole extension |
| `COMMANDS` | `+ UAL_COMMANDS` | new subcommands |
| `build_parser()` | `--dataset` and the `--ual_*` flags appended | dataset selector and UAL options |
| `main()` | three lines at the top: `*_ual` → `main_ual`; legacy commands refuse `--dataset ual_bench` | routing and guard |

No existing function body, default, prompt, path or cache key was changed. The
legacy `selftest` prints byte-identical output before and after the patch
(59/59; temporary paths aside).

## Schema audit (verified in the official repository)

Source: `https://github.com/Hasnat79/UAL_Bench` (cloned; annotation JSON files are
in `data/`, one sample video per subset). Hugging Face and the Drive folders were
**not reachable** from the build environment, so the videos and the Drive copies of
the annotation files were not inspected.

| | UAG-OOPS | UAG-SSBD | UAG-FunQA |
|---|---|---|---|
| file | `data/uag_oops/oops_uag_paper_version.json` | `data/uag_ssbd/ssbd_paper_version.json` | `data/uag_funqa/uag_funqa_dataset.json` |
| sha256 | `a5e92a7c…a495b1` | `f3b413d2…3c3996` | `6b7b6478…0e10c59` |
| structure | dict `{video_id: {start_time, end_time, description}}` | list `[[video_tag, {id, time, bodypart, category, intensity, modality, description}]]` | list `[{instruction, visual_input, output, task, ID, start_time, end_time}]` |
| items / videos | 1589 / 1589 | **104 / 58** (21 videos have ≥2 behaviours, max 7) | 172 / 172 |
| evaluation id | dict key | `video_tag + "_" + id` (`ssbd_loader.py`) | `visual_input` (key of the official prediction files) |
| query (official scripts) | `description` | `f"A person is {category} with {intensity} intensity."` | `instruction` |
| distinct queries | 1572 (17 items are literally `"Don't know Don't know"`) | 6 | **1** — `"Identify the video's funny moment."` |
| GT units | seconds (float) | `time` = `"MMSS:MMSS"` → seconds (see below) | seconds = `round(frame/30, 2)` of `output` (verified for all 172) |
| GT length mean / median | 5.00 / 4.17 s | 17.9 / 10 s | 4.31 / 3.20 s |
| zero-length / reversed GT | 0 / 0 | 0 / 0 | 0 / 0 |
| splits | none (evaluation only) | none | none |

**UAG-OOPS end-time convention.** All 1589 `end_time` values are multiples of
1/8 s, but only 56 of the start times are. So `end_time` is the clip end, not an annotated event end.
Concrete record: `"34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0"`,
`end_time = 3.75`, while the mp4 lasts 3.637 s (header frames/fps; ffprobe stream
3.637 s, container 3.658 s). The official evaluator uses `duration = gt_end`
(`eval_your_results.py`, lines 24, 49, 77). The script keeps `end_time` verbatim. Predictions are
bounded by the decoded video, so an "until the end" prediction keeps an
unavoidable end error of up to ~0.1 s on such clips. Nothing is repaired.

**UAG-SSBD time format.** `data/README.md` says "start and end time (seconds)", but
the file stores `MMSS`: 38 records have a field ≥ `0100` and no seconds field reaches
60, which is implausible for plain seconds. The repository has no conversion code, so
this is a protocol ambiguity. Default `--ual_ssbd_time_format mmss`
(`'0105:0125'` → 65 s to 85 s); `seconds` is available. With the videos present,
`inspect_ual` counts intervals that would end after the video under each reading.
Run it before reporting SSBD numbers. With the single sample video available here
(`v_ArmFlapping_01`, 45.5 s; times 18–24 s and 27–32 s) both readings coincide, so
the choice is **not yet verified on data**.

**Training data.** `uag_oops_train_instruct_dataset_v1.json` (Drive name) and
`data/uag_oops_train_instruct/uag_oops_train_instruct_dataset.json` (repository name)
are rejected by name, by directory and by schema (`[{video, QA}]`).

## Official metrics (`src/eval/eval_your_results.py`, ported exactly)

* `IoU = max(0, min(ge,pe) − max(gs,ps)) / ((ge−gs) + (pe−ps) − inter)` — no clipping.
* `TD = |gs − ps| + |ge − pe|` — the **sum** of both boundary errors.
* Null prediction (`pred_start` or `pred_end` is `None`): IoU = 0, TD = `gt end_time`.
* `R@1,IoU≥m`, m ∈ {0.3, 0.5, 0.7}; `R@1,TD≤p`, p ∈ {0, 1, 3, 5, 7}: **percentages** (100 × correct / N).
* `mIoU`: fraction in [0, 1]. `mTD`: seconds; a TD > 10000 enters the mean as `gt end_time`.
* Onset accuracy `|gs − ps| ≤ 1 s` and `≤ 0.25 s` **is** implemented in the official
  evaluator (`accuracy_pred_start`), so it is reported in the supplementary table.
* Prediction schema (`src/eval/output_test_template.json`): `[{video_id, start_time, end_time, pred_start, pred_end}]`.

`selftest_ual --ual_repo <clone>` runs the unmodified official script and requires
the same printed values on the template (README numbers), on 300 random items
with null predictions, and on every end-to-end fixture run. `run_ual`/`evaluate_ual
--ual_repo <clone>` repeat that check on every real predictions file.

## Mode A — the primary experiment (query-conditioned CRUX localisation core)

For each item, using only the query, the windows and the decoded duration:

1. **Windows**: the unchanged GliDe geometry of `prepare_record` (4 fps, 8-frame stitched windows = 2 s).
2. **V**: `s_q(w) = logit P(Yes | w, q) − logit P(Yes | grey, q)` — `verify_profile` with the
   official query as the hypothesis (contextual calibration, Zhao et al., 2021).
3. **S**: `segment_profile(s_q, τ, λ)` with τ = 0, λ = 10⁹. This is the K = 1 case of the CRUX
   segmentation: the single interval maximising Σ(s_q − τ) (Kadane). It always returns
   one interval, which is what the official protocol scores.
4. **Frame refinement**: masked-frame bisection on the two boundary windows with the presence
   oracle at margin τ (`frame_onset`, `frame_offset`, `frame_time_bounds`, unchanged).
5. The interval is clipped to [0, decoded duration]. Ground truth is never read in this path.

**Not used:** H (open glitch descriptions, event resolution, summariser) and U (utility
selector fitted on VideoGlitchBench judge ratings). This is **not** an evaluation of the
full H → V → S → U pipeline. **Adapter:** the presence prompt. `P_PRESENCE_SYS` names game
glitches, so `P_UAL_PRESENCE_SYS` / `ual_presence_user` swap in a generic "event" wording
and add the frame-layout sentence of `P_DETECT_SYS`. It is the same prompt for all three
subsets. **No calibration**: UAL-Bench has no dev split, so τ, λ and the refine mode were
fixed before any UAL result was seen. Changing them with `--ual_seg_tau / --ual_seg_penalty /
--ual_refine_mode` is flagged in the manifest and summary as a deviation.

**Mode B** (`--ual_mode open_ended_full_pipeline`) is intentionally out of scope and
refused. H and U are glitch-specific and VideoGlitchBench-fitted, so running them here
would be a different method that would also need a validated query-matching rule.

## Commands

```bash
# 0. tests (no model, no GPU); --ual_repo adds parity with the official evaluator
python crux_v10.py selftest
python crux_v10.py selftest_ual --ual_repo /path/UAL_Bench

# 1. inspect annotations + media (no VLM)
python crux_v10.py inspect_ual --ual_root /path/UAL_Bench --ual_dataset all

# 2. serve each approved checkpoint (one OpenAI-compatible server per backbone)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct   --port 8000 --max-logprobs 20
vllm serve OpenGVLab/InternVL2_5-8B      --port 8002 --max-logprobs 20 --trust-remote-code
vllm serve ByteDance-Seed/UI-TARS-1.5-7B --port 8003 --max-logprobs 20
API="qwen2.5-vl-7b=http://localhost:8000/v1,internvl2.5-8b=http://localhost:8002/v1,ui-tars-1.5-7b=http://localhost:8003/v1"

# 3. one-sample smoke test (marked PARTIAL everywhere)
python crux_v10.py run_ual --ual_root /path/UAL_Bench --ual_dataset uag_oops \
    --backbone qwen2.5-vl-7b --limit 1 --ual_api_bases "$API"

# 4. each subset separately
for d in uag_oops uag_ssbd uag_funqa; do
  python crux_v10.py run_ual --ual_root /path/UAL_Bench --ual_dataset $d \
      --backbone qwen2.5-vl-7b --ual_api_bases "$API" --ual_repo /path/UAL_Bench
done

# 5. all three subsets x all three approved backbones
python crux_v10.py run_ual --ual_root /path/UAL_Bench --ual_dataset all --backbone all \
    --ual_api_bases "$API" --ual_repo /path/UAL_Bench

# 6. rebuild every table from saved runs (no model calls)
python crux_v10.py evaluate_ual --ual_root /path/UAL_Bench --ual_dataset all --backbone all \
    --ual_repo /path/UAL_Bench

# score any file in the official prediction schema
python crux_v10.py evaluate_ual --ual_predictions predictions_official.json --ual_repo /path/UAL_Bench
```

Video lookup order per subset: `--ual_<subset>_videos`; then `<root>/data/<subset>/videos`,
`<root>/<subset>/videos`, `<root>/<subset>` for `--ual_video_root` and then `--ual_root`.
Annotation lookup: `--ual_<subset>_annotations` / `--ual_annotations` (single subset);
then `<ual_root>/data/<subset>/<file>`, `<ual_root>/<subset>/<file>`, `<ual_root>/<file>`.
Missing videos abort before any inference unless `--ual_allow_missing_videos` is passed;
those items are then scored as official null predictions.

`--backend hf` runs Qwen2.5-VL-7B and UI-TARS-1.5-7B; their architecture is checked to be
`Qwen2_5_VLForConditionalGeneration`. InternVL2.5-8B needs vLLM (its remote code hides the
logits). Mixing backends across backbones is not supported, so all backbones of one call use
the same `--backend`. `max_pixels`, fps, window size and the prompt are identical for all
backbones and recorded in every manifest.

## Output layout

```
<parent of --ual_root>/crux_v10_ual_work/        (or --ual_work_root)
└── ual_bench/
    ├── logs/crux_v10_<command>.log
    ├── inspect/schema_audit.json
    ├── <subset>/
    │   ├── windows/<prep fingerprint>/<video key>/   stitched windows (shared by backbones)
    │   ├── cache/<backbone>/                          VLM cache, keys namespaced by
    │   │                                              dataset, input fingerprint, prompt,
    │   │                                              adapter, model, sampling (GT-free)
    │   └── <backbone>/query_conditioned/<full|partial_limitN>/
    │       ├── raw_outputs.json          per item: query, status, profile, segments,
    │       │                             pred_start/pred_end, calls, truncations (no GT)
    │       ├── predictions_official.json official schema (joined with GT here only)
    │       ├── metrics.json              official metrics, parity, diagnostics
    │       └── manifest.json             config, versions, hashes, preflight, smoke test
    └── summary/query_conditioned/<full|partial_limitN>/<subsets>__<backbones>/
        ├── results.md   results.csv   results.json
```

The work root may not lie inside the VideoGlitchBench root (`--root`), `crux_v8_work`,
the GliDe repository, the UAL-Bench data root or any video/annotation directory.

Example `predictions_official.json` entry (from a run on the real SSBD sample video):

```json
{"video_id": "v_ArmFlapping_01_b_01", "start_time": 18.0, "end_time": 24.0,
 "pred_start": 34.0, "pred_end": 36.0}
```

(The prediction comes from a content-blind placeholder used only to exercise real-video
decoding. It is not a result.)

## Results

Main table header (one table per subset in `results.md`, one row per backbone):

`Model | R@1,0.3 | R@1,0.5 | R@1,0.7 | R@1,TD≤0 | R@1,TD≤1 | R@1,TD≤3 | R@1,TD≤5 | mTD`

**No results have been measured yet.** No GPU, model weights or UAL-Bench videos were
available in the build environment. `results.md` also contains a supplementary table of
official metrics (mIoU, R@1,TD≤7, onset accuracy) and non-official diagnostics. The
diagnostics are null predictions, Yes/No top-logprob truncation, calls per item,
video-cluster bootstrap CIs, a whole-video reference span, and paired backbone
sign-flip tests with Holm correction.

## Limitations

* Videos, Hugging Face and the Drive folders could not be accessed from the build
  environment. Media resolution, the OOPS end-time rule on all clips and the SSBD time
  format must be confirmed with `inspect_ual` on the downloaded data.
* The published UAL-Bench tables could not be retrieved, so no published number is
  quoted. Published systems (Vid-LLMs, VLM-LLM pipelines) differ from CRUX, and any
  comparison must name the subset, metric and protocol. Nothing here supports a SOTA claim.
* UAG-FunQA has one constant query, so Mode A there is query-conditioned in name only.
  17 UAG-OOPS items have the uninformative query "Don't know Don't know". Both are kept
  as the official protocol defines them.
* A model whose first token is not Yes/No (possible for UI-TARS, a GUI-agent model)
  gets floored log-probs. The truncation share is reported per backbone and must be
  read before comparing backbones.
* Preprocessing keeps the GliDe sampler exactly. Its last seek can fail at a clip end,
  so predictions stop at the last decoded frame + 0.25 s.
