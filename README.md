# XGXW-E1: Counseling Quality Evaluation of a Tri-Agent Parliament on Real Help-Seeking Posts

Anonymous code & data release for the paper *"Knowledge-Guided Multi-Agent Deliberation Improves Counseling Quality of LLM Mental-Health Support"* (under review).

## Contents

```
e1_counseling_quality/   # E1 pipeline (sampling, generation, dual-rubric blind judging, statistics, figures)
  sample_corpus.py       #   stratified sampling from PsyQA / CPsyCounD
  conditions.py          #   8 experimental conditions & backend callers
  run_generation.py      #   batch generation (resumable, shardable)
  run_judge.py           #   dual-rubric blind judging (--rubric autocbt|conv|both)
  analyze_results.py     #   Welch t / Cohen's d / Holm + length-bias analysis
  make_figures.py        #   pure-Python SVG figures
e2_multiturn/            # E2 multi-turn process-evaluation pipeline (virtual help-seeker; pending compute)
e1_corpus/               # fixed stratified samples (seed=42; 200 PsyQA posts + 100 CPsyCounD sessions)
e1_results/
  generations/           # responses of all 8 conditions (N=50 each), JSONL
  judgments/             # AutoCBT coverage-rubric blind scores
  judgments_conv/        # dialogue-appropriateness rubric blind scores
  E1_report_*.md         # full statistical report
  E1_analysis_*.json     # machine-readable statistics
  figures/               # SVG figures (incl. dual-rubric comparison)
```

## Key results

- Rubric-dependent reversal: single-model long-form wins on coverage (35.6 vs 23.4 / 42), while the tri-agent parliament wins decisively on dialogue appropriateness (36.2 vs 16.4–18.6 / 42).
- Length bias: coverage-rubric score correlates with response length (Pearson r=0.60); appropriateness rubric correlates negatively (r=−0.65).
- The parliament exceeds human online peer replies under both rubrics (Cohen's d=1.35, p<0.001) with near-zero safety-violation rates.

## Reproduce

1. Download raw corpora (not redistributed here due to size/licensing):
   - PsyQA: `lsy641/PsyQA` (HuggingFace)
   - CPsyCounD: `CAS-SIAT-XinHai/CPsyCoun` (HuggingFace)
   Place under `e1_corpus/raw/`.
2. Configure an OpenAI-compatible endpoint for generation (paper: qwen3.7-max) and judging (paper: glm-5.2) via env vars `LLM_API_URL/LLM_API_KEY/LLM_MODEL` and `E1_JUDGE_API_URL/E1_JUDGE_API_KEY/E1_JUDGE_MODEL`.
3. `python sample_corpus.py && python run_generation.py --condition <cond> && python run_judge.py --rubric both && python analyze_results.py && python make_figures.py`

GVC conditions additionally require the full platform (multi-agent service); its code will be released upon publication. All judged outputs needed to verify the paper's numbers are already included in `e1_results/`.

## License / Ethics

Research use only. Source corpora retain their original licenses (PsyQA: research license; CPsyCounD: CC BY-SA 4.0). No personally identifiable information is included; help-seeking posts come from the public PsyQA dataset.
