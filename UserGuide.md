# Micro LLM Creator User Guide

This guide explains how to create a small language model from scratch with
Micro LLM Creator. The app is designed for local experiments: preparing text or
programming data, training a small GPT-style model, resuming interrupted runs,
and exporting the result.

## 1. Install and Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the app:

```powershell
python run_app.py
```

On Linux/macOS, use:

```bash
python3 run_app.py
```

If you want to launch it directly as `./run_app.py`, mark it executable once:

```bash
chmod +x run_app.py
./run_app.py
```

Minimum supported Python version is Python 3.9.

The app has five main work areas:

- `IN`: prepare datasets.
- `AI`: configure and train the model.
- `Bench`: run repeatable benchmark prompts against a trained checkpoint.
- `X`: export and quantize model artifacts.
- `Chat`: load a GGUF model and chat with it locally.

## 2. Recommended Workflow

1. Collect clean base training material.
2. Prepare a `Base pretraining` dataset from local files and optional base corpora.
3. Open `AI`, choose `Pretrain from scratch`, and train the base checkpoint.
4. Prepare an `Instruction fine-tune` dataset if you want request-following behavior.
5. Choose `Instruction fine-tune`, select the base checkpoint, and fine-tune.
6. Prepare a `Conversation fine-tune` dataset if you want assistant/chat behavior.
7. Choose `Conversation fine-tune`, select the latest compatible checkpoint, and fine-tune.
8. Resume training if interrupted.
9. Open `X` to bundle, quantize, or convert the trained model.
10. Open `Chat`, load a GGUF model, and test prompts.
11. Use `Save Project` to store paths and settings for the next session.

### Training Stages

Micro LLM Creator separates model creation into stages.

`Base pretraining`

- Starts from random weights.
- Learns tokenizer usage, grammar, language patterns, code syntax, and broad
  next-token prediction.
- Use local PDFs, text, code files, and optional base corpora.
- Online base corpora are opt-in. Nothing is selected by default.

Optional online base datasets:

- `TinyStories`: simple short stories for basic fluency.
- `WikiText-103`: clean Wikipedia-style long-form text.
- `Wikipedia EN 2023`: broad encyclopedia prose. Use a row limit.
- `FineWeb-Edu sample`: large educational web text. Use a row limit.

`Instruction fine-tune`

- Loads a compatible base checkpoint.
- Starts a fresh optimizer run.
- Teaches the model to follow tasks and produce answer-shaped responses.
- Use datasets such as Alpaca, Dolly, or SlimOrca.

`Conversation fine-tune`

- Loads a compatible base or instruction-tuned checkpoint.
- Teaches multi-turn chat style, short replies, helpfulness, and conversational
  flow.
- Use datasets such as UltraChat, DailyDialog, or OpenAssistant.

TinyStories is not selected by default and is not shown in fine-tuning stages.
It is a base-pretraining option only.

## 2.1 New, Save, and Open Projects

The top bar has a project name field plus `New Project`, `Save Project`, and
`Open Project`.

`New Project` clears the active project binding and restores fresh defaults.
Use it when you opened an existing project but want to start a different one.
It resets visible status, progress bars, logs, charts, chat state, and default
run folders. The next `Save Project` will ask for a new parent folder.

`Save Project` creates a folder using the project name and writes a
`project.json` file inside it. This file stores:

- Source, dataset, model, export, GGUF, tokenizer, and checkpoint paths.
- Dataset preparation options.
- Tokenizer policy.
- Training architecture and optimizer options.
- Export and chat settings.
- Small summaries from existing dataset/model folders when available.

On first save, the app also creates a standard project workspace:

```text
YourProject/
  project.json
  datasets/
  models/
  exports/
  cache/
  temp/
```

The app then points these fields at the project folders:

- Dataset Core -> `datasets/`
- Training Dataset -> `datasets/`
- Model Output -> `models/`
- Export Model Core -> `models/`
- Output Bay -> `exports/`
- GGUF Output -> `exports/model.gguf`

The `cache/` and `temp/` folders are used as preferred runtime locations for
libraries that honor Python/PyTorch/Hugging Face cache and temp environment
variables.

Important:

- The project file stores paths to large assets; it does not duplicate hundreds
  of PDFs or large model checkpoints.
- Keep your dataset and model folders in stable locations if you want projects
  to reopen cleanly.
- Use `Open Project` to restore the app controls from a saved `project.json`.
- Use `New Project` before creating a separate experiment from scratch.

### Disk Space During Training

Micro LLM Creator writes prepared datasets and checkpoints to the paths shown in
the app. Check these first when estimating disk use:

- `datasets/`: corpus, tokenizer, token files, extraction cache, versions.
- `models/`: checkpoints, final model, summaries, lineage.
- `exports/`: bundles, FP16 checkpoints, GGUF outputs.

Some C drive usage can still happen outside the app:

- Windows `%TEMP%` and Python temporary files.
- PyTorch/CUDA runtime caches.
- pip package cache or installed packages.
- OS page file or memory compression while training.
- GPU/driver shader/kernel caches.

After a project is saved or opened, the app prefers project-local `cache/` and
`temp/` folders for common runtime cache/temp variables. Windows and GPU drivers
may still use system locations for some low-level work.

## 3. Dataset Preparation

Dataset preparation converts your files into a tokenizer-ready corpus. It also
trains a byte-level BPE tokenizer and splits tokens into training and validation
streams.

The `IN` tab is organized into:

- `Source Array`: source folder, dataset folder, parallel workers, and prepare mode.
- `Tokenizer Core`: vocabulary, tokenizer policy, context, validation, and code options.
- `Dataset Quality`: sample, token, vocabulary, code/prose, cache, and warning summary.
- `Dataset Advisor`: visible cleanup suggestions after Preview Dataset.
- `Ingest Telemetry`: live preparation messages while files are processed.

### Check Health

Runs a project readiness check before long work.

It validates:

- source folder and supported files
- prepared dataset artifacts
- trained model/checkpoint artifacts
- export folder
- selected GGUF chat model
- llama.cpp converter path when provided
- selected training device and CUDA availability

Effect:

- Helps catch missing paths before preparing or training.
- Warns when a dataset/model/export is not created yet.
- Reports an error if CUDA is selected but PyTorch cannot use CUDA.

### Preview Dataset

Scans the source folder and prepared dataset state without writing training
artifacts.

It shows:

- supported source file count
- source size
- file type breakdown
- prepared dataset status
- likely duplicate files or repeated extracted text
- suspicious extraction quality
- code/prose balance
- training readiness score
- quality notes
- a few readable text/code previews

Effect:

- Lets you inspect whether PDFs/text/code are extracting cleanly.
- Helps catch tiny datasets, unreadable files, and incomplete prepared folders.
- Helps catch duplicate-heavy datasets before the model memorizes repeated text.
- Gives a simple readiness signal before you spend training time.
- Gives confidence before spending time on full dataset preparation.

### Duplicate Detection

Preview Dataset checks for:

- exact duplicate files with identical content
- repeated extracted text previews

Effect on the LLM:

- Duplicate-heavy datasets make small models memorize repeated passages.
- Duplicates can make training loss look better while validation/generation
  quality stays weak.
- Remove duplicate books, repeated exports, copied folders, and generated files
  before serious training.

### Bad Extraction Detection

Preview Dataset flags source files with suspicious text extraction.

Examples:

- PDFs that produce almost no readable text.
- Text with very high symbol/noise ratio.
- Long repeated character runs.
- Encoding artifacts.
- Very low word variety.

Effect on the LLM:

- Bad PDF extraction teaches broken spacing, broken code, and noisy symbols.
- Noisy files waste model capacity.
- Remove or replace flagged files, or use cleaner source formats when possible.

### Code/Prose Balance

Preview Dataset reports whether the source looks prose-heavy, code-heavy, or
balanced.

Effect on the LLM:

- Prose-heavy datasets are better for explanation but weaker for code syntax.
- Code-heavy datasets are better for syntax but may give terse explanations.
- Balanced code/prose is usually best for a programming assistant that writes
  and explains code.

### Training Readiness

Preview Dataset gives a score from `0` to `100` with a label:

- `Ready`
- `Usable with warnings`
- `Needs cleanup`
- `Not ready`

The score considers:

- prepared token count when available
- source size before preparation
- duplicate ratio
- bad extraction ratio
- code/prose balance for code-training projects

Use the reasons in Ingest Telemetry to decide what to fix before training.

### Source Vault

The source folder containing your documents.

Supported document files:

- `.pdf`
- `.txt`
- `.md`
- `.text`
- `.jsonl`

When Code Training Mode is enabled, source-code files are also included:

- `.py`, `.js`, `.ts`, `.jsx`, `.tsx`
- `.java`, `.c`, `.cpp`, `.h`, `.hpp`
- `.cs`, `.go`, `.rs`, `.php`, `.rb`
- `.swift`, `.kt`, `.scala`, `.r`
- `.sql`, `.sh`, `.ps1`
- `.html`, `.css`, `.xml`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`

Effect on the LLM:

- More high-quality data improves coverage and fluency.
- Badly extracted PDFs can teach broken formatting.
- Real source-code files are much better than code copied from PDFs.

### Dataset Purpose

Selects the role of the prepared dataset.

- `Base pretraining`: base language/model training from local files and optional
  base corpora.
- `Instruction fine-tune`: task-following data for a checkpoint that already
  knows language basics.
- `Conversation fine-tune`: multi-turn chat data for assistant behavior.

Effect on the LLM:

- Base pretraining teaches broad prediction ability.
- Instruction fine-tuning teaches prompt/answer behavior.
- Conversation fine-tuning teaches chat style and turn-taking.

The selected purpose also filters the online dataset list. TinyStories appears
only in `Base pretraining`; instruction and chat datasets appear only in their
fine-tuning stages.

### Include Online Training Datasets

Downloads or reads selected Hugging Face datasets into the project dataset
cache. This is opt-in. No online dataset is selected by default.

Base pretraining options:

- `TinyStories`: basic language fluency.
- `WikiText-103`: clean long-form text.
- `Wikipedia EN 2023`: broad encyclopedia prose.
- `FineWeb-Edu sample`: educational web text.

Instruction fine-tune options:

- `Alpaca 52K`: compact instruction-following examples.
- `Dolly 15K`: human-written instruction examples.
- `SlimOrca`: instruction and reasoning-style assistant answers.

Conversation fine-tune options:

- `UltraChat 200K`: multi-turn assistant conversations.
- `DailyDialog`: everyday dialogue.
- `OpenAssistant OASST1`: assistant-style conversation messages.

Use `Rows per dataset` to limit large downloads. Start with a small limit for a
smoke test before preparing a large dataset.

### Dataset Core

The output folder for prepared artifacts.

The app writes:

- `corpus.txt`
- `tokenizer.json`
- `train_tokens.json`
- `val_tokens.json`
- `dataset_summary.json`
- `dataset_lineage.json`
- `versions/<version_id>/dataset_summary.json`
- `versions/<version_id>/dataset_manifest.json`

Effect on the LLM:

- This folder becomes the training source for the `AI` tab.
- Reusing the same dataset keeps experiments comparable.
- Each preparation run records a dataset version so you can trace which data
  produced which model.

### Dataset Versions

Every successful preparation creates a dataset version such as:

```text
v001_20260629T120000Z_a1b2c3d4e5f6
```

The version records:

- source file hashes
- preparation settings
- tokenizer policy
- tokenizer SHA-256 hash
- token counts
- code/prose counts
- manifest snapshot

When a model is trained, the training output records the dataset version in
`model_lineage.json` and `training_summary.json`. This is important because it
lets you answer: "Which exact data produced this checkpoint?"

### Parallel Lanes

Number of files read in parallel.

Effect:

- Higher values can speed up hundreds of PDFs/text files.
- Very high values can make disk usage and CPU load heavy.
- A good starting value is `4` to `8`.

### Prepare Mode

Controls how the dataset is updated.

- `Incremental update`: reuses cached extracted text for unchanged files, processes only new/changed files, and can reuse the existing tokenizer when available.
- `Full rebuild`: rebuilds corpus, tokenizer, and token files from all current source files. Cached extraction can still avoid rereading unchanged PDFs.
- `Force reprocess`: ignores extraction cache, rereads all source files, rebuilds tokenizer, and rewrites token files.

Recommendation:

- Use `Incremental update` when you add more PDFs/source files later.
- Use `Full rebuild` when you want the tokenizer to learn from all data again.
- Use `Force reprocess` if extracted text looks wrong or source parsing options changed.

The app writes `dataset_manifest.json` and cached extracted samples under
`cache/documents` in the dataset folder.

### Tokenizer Policy

Controls how `tokenizer.json` is created.

- `Auto`: recommended default. During incremental updates, the app reuses the
  existing dataset tokenizer when it exists. Otherwise, it trains a new
  tokenizer from the corpus.
- `Train new tokenizer`: always trains a fresh tokenizer from the current
  corpus. Use this after major corpus changes when you want the vocabulary to
  relearn all data.
- `Reuse dataset tokenizer`: requires an existing `tokenizer.json` in the
  dataset folder. Use this when adding more data to an already trained model
  family so token IDs stay stable.
- `Import tokenizer.json`: copies a tokenizer from another compatible project
  into the dataset folder. Use this only when the new dataset should stay
  compatible with that tokenizer.

Effect on the LLM:

- Reusing a tokenizer keeps token IDs stable, which is important when continuing
  work from older checkpoints.
- Training a fresh tokenizer can better fit a changed corpus, but old
  checkpoints are no longer compatible because token IDs may change.
- Imported tokenizers are useful for professional workflows where multiple
  datasets share the same vocabulary.

Compatibility requirement:

- A training tokenizer must contain `<pad>`, `<unk>`, `<bos>`, and `<eos>`.
- The app validates imported/reused tokenizers during dataset preparation.
- The app records `tokenizer_sha256` in dataset and model lineage.

### Lowercase Text

Converts text to lowercase.

Effect:

- Reduces vocabulary pressure.
- Loses capitalization patterns.
- Usually keep off for code, because case matters in many languages.

### Code Training Mode

Enables code-aware preparation.

Effect:

- Keeps code formatting.
- Tags code and prose separately.
- Includes source-code files.
- Extracts code-like blocks from PDFs/text.

Recommendation:

- Enable this for programming books, tutorials, and source repositories.

### Include Source Files

Includes real source-code files from the source folder.

Effect:

- Strongly improves code syntax learning.
- Preserves real project structure and idioms.
- Better than relying only on PDF-extracted code.

Recommendation:

- Keep enabled when training for code.

### Include Explanations

Keeps prose from books, PDFs, and tutorials.

Effect:

- Helps the model learn concepts, descriptions, and explanatory language.
- Useful for "explain this code" behavior.
- Too much prose without code can reduce code density.

Recommendation:

- Keep enabled for programming assistant behavior.
- Disable only if you want a mostly syntax/code-completion model.

### Extract Code Blocks

Attempts to detect code-like sections inside PDFs/text.

Effect:

- Helps recover examples from programming PDFs.
- Detection is heuristic; some blocks may be missed or noisy.
- Real source files remain more reliable.

Recommendation:

- Keep enabled for programming PDFs.

### Preserve Indentation

Keeps line breaks and indentation in code.

Effect:

- Critical for Python.
- Improves readability and generated code structure.
- Avoids flattening code into broken single-line text.

Recommendation:

- Keep enabled for all code datasets.

### Instruction-Style Samples

Wraps code in simple instruction tags.

Example:

````text
<sample type="reasoning_code" language="python" source="example.py">
<instruction>Write or explain the python code for example.</instruction>
<reasoning>
Understand the requested programming task, choose the relevant language patterns,
preserve correct syntax, and provide the implementation.
</reasoning>
<answer>
```python
def hello():
    return "hi"
```
</answer>
<explanation>The answer contains the implementation that satisfies the task.</explanation>
</sample>
````

Effect:

- Gives the model a hint that code is a task-oriented sample.
- Useful for instruction-style prompting later.

Recommendation:

- Keep enabled for general coding assistant behavior.

### Reasoning Samples

Controls how code instruction samples are shaped.

- `Reasoning scaffold`: adds short task, reasoning, answer, and explanation
  sections. This is the default.
- `Detailed code reasoning`: adds a more explicit checklist around goal,
  inputs, outputs, control flow, data structures, syntax, and edge cases.
- `No reasoning wrapper`: keeps a simpler instruction plus answer format.

Effect:

- Helps the model learn a response structure similar to coding assistants.
- Helps prompts like "explain", "review", "fix", or "write code" produce more
  organized answers.
- Does not magically create deep reasoning; for that, you still need many
  high-quality examples with real problem-solving traces.

Recommendation:

- Use `Reasoning scaffold` for most code datasets.
- Use `Detailed code reasoning` when training specifically for explanation,
  debugging, and code review behavior.

### Auto Vocabulary

Lets the app estimate tokenizer vocabulary size.

Effect:

- Safer default for beginners.
- Prevents tiny corpora from using unnecessarily huge vocabularies.
- Larger corpora automatically receive larger vocabulary suggestions.

Recommendation:

- Keep enabled unless you are comparing tokenizer experiments.

### Manual Vocabulary

Manual target vocabulary size.

Effect:

- Larger vocabulary can preserve more words/symbol patterns.
- Larger vocabulary increases model output layer size.
- Too large for a small dataset wastes model capacity.

Rules of thumb:

- Tiny experiments: `512` to `2,000`
- Small serious datasets: `4,000` to `8,000`
- Larger mixed code/prose datasets: `16,000` to `32,000`

### Minimum Frequency

Minimum frequency for tokenizer tokens.

Effect:

- Higher values remove rare fragments.
- Lower values preserve rare symbols/names.

Recommendation:

- Use `2` as a balanced default.
- Use `1` for code-heavy datasets with many rare identifiers.

### Context Window

Number of tokens per training sequence.

Effect:

- Larger context lets the model learn longer dependencies.
- Larger context uses more memory.
- For code, longer context helps functions/classes stay coherent.

Starting values:

- CPU/tiny test: `64` to `128`
- Small GPU training: `128` to `512`
- Larger GPU experiments: `1024+`

### Validation Split

Fraction of data held out for validation.

Effect:

- Validation loss helps detect overfitting.
- Too much validation leaves less training data.

Recommendation:

- Use `0.1` for most datasets.
- Use `0.05` for very small datasets.

## 4. Training Options

Training uses next-token prediction: the model sees tokens and learns to predict
the next token.

Before training starts, the app runs a pre-training checklist. It validates the
prepared dataset, tokenizer, train/validation token files, model architecture,
selected device, resume checkpoint, estimated checkpoint size, and free disk
space on the model output drive.

Hard errors block training. Warnings are shown in Training Telemetry and ask for
confirmation before continuing.

The checklist also shows:

- estimated parameter count
- estimated checkpoint size
- rough VRAM estimate
- estimated total training storage
- free model-drive space
- run history count

Every completed or safely stopped training run is appended to
`models/training_history.json`. The history records checkpoint path, losses,
dataset path/version, run id, model config, and training config.

The `AI` tab also has a `Model Estimate` card. Click `Refresh Estimate` to
update parameter count, checkpoint size, rough VRAM, and run-history count
without starting training.

### Dataset Project

Folder created by dataset preparation.

Must contain:

- `tokenizer.json`
- `train_tokens.json`
- `val_tokens.json`

### Model Output

Folder where training outputs are saved.

Includes:

- `final_model.pt`
- `tokenizer.json`
- `training_summary.json`
- `model_lineage.json`
- `checkpoints/`

`model_lineage.json` records the training run ID, source dataset folder,
dataset ID, dataset version, tokenizer size, resume checkpoint, compatibility
safety setting, and checkpoint path.

### Resume Safety

`Resume latest` continues from the newest checkpoint in the model output
folder. You can also choose a specific checkpoint path.

Click `Check Resume` to preview compatibility before starting training. The
report appears in the `Resume Compatibility` panel and shows blocking errors,
warnings, and safe-resume details.

`Safe resume` checks the checkpoint before training starts. It blocks resume
when the tokenizer or model shape changed, including vocabulary size, context
length, `n_embd`, `n_head`, `n_layer`, block style, RoPE settings, attention
type, or effective KV head count. These changes make checkpoint weights
incompatible.

When `Safe resume` is enabled, optimizer, scheduler, and AMP scaler state must
also match so the run continues exactly. If you intentionally want to continue
from model weights with a different optimizer or schedule, disable `Safe
resume`. The app will then skip incompatible optimizer/scheduler/scaler state
and continue as a weight-only fine-tuning run.

Changes such as dropout, sliding attention window, attention backend, learning
rate, weight decay, and gradient clipping are shown as warnings because they
change training behavior without necessarily changing checkpoint tensor shapes.

### Training Mode

`Pretrain from scratch` starts from random model weights. Use this for the first
model in a model family.

`Instruction fine-tune` loads compatible model weights from a base checkpoint,
then starts a fresh optimizer and scheduler run on an instruction dataset. Use
this after base pretraining when you want the model to follow requests.

`Conversation fine-tune` loads compatible model weights from a base or
instruction-tuned checkpoint, then trains on multi-turn dialogue/chat data. Use
this after the model has learned basic language patterns.

`Fine-tune checkpoint` is the generic fine-tune mode for custom/domain data
that is not specifically instruction or conversation data.

Click `Check Fine-tune` before training. The compatibility report appears in the
same `Resume Compatibility` panel.

Important:

- Fine-tuning requires the same tokenizer vocabulary and compatible model shape.
- Fine-tuning does not load optimizer, scheduler, or AMP scaler state.
- Resuming an existing run takes priority if `Resume latest` finds a checkpoint
  in the current model output folder.
- For fine-tuning, prepare the new dataset with `Reuse dataset tokenizer` or
  `Import tokenizer.json` from the base model family so token IDs remain stable.

### PEFT / LoRA

`Full fine-tune` updates all trainable model weights.

`LoRA adapters` freezes the base model and trains small low-rank adapter
matrices on selected projection layers. This is parameter-efficient fine-tuning:
intermediate checkpoints store only adapter weights, optimizer state, scheduler
state, and metadata instead of rewriting the whole model every time.

LoRA outputs:

- `checkpoints/checkpoint_<step>.pt`: adapter-only resumable checkpoint.
- `checkpoints/checkpoint_stopped_step_<step>.pt`: adapter-only stopped
  checkpoint.
- `final_adapter.pt`: final adapter-only checkpoint.
- `final_model.pt`: merged full checkpoint for existing benchmark/export tools.

LoRA options:

- `LoRA rank`: adapter capacity. Higher is stronger but larger.
- `LoRA alpha`: adapter scaling. A common starting point is `2 * rank`.
- `LoRA dropout`: adapter-only dropout.
- `LoRA target`: attach adapters to attention projections, MLP projections, or
  both.

Recommendation:

- Use `LoRA adapters` for most task adaptation and code fine-tuning.
- Use `Full fine-tune` when you intentionally want every model weight updated.
- Keep a stable base checkpoint and tokenizer for a model family.

### Training Profile

Profiles quickly apply practical optimizer, schedule, precision, and
regularization choices.

- `Stable LLM`: AdamW, cosine schedule, normal defaults.
- `Low-memory`: Adafactor, grouped-query attention, lower memory profile.
- `Code fine-tune`: lower learning rate and lower gradient clip for adapting a
  base checkpoint to code data.
- `Experimental Lion`: Lion optimizer with one-cycle schedule for experiments.

Profiles are a starting point. You can still edit every option after applying a
profile.

### Preset

Quick architecture presets.

- `Tiny`: faster, lower quality, good for testing.
- `Small`: more capacity, needs more data and memory.
- `Custom`: use your own values.

### Block Style

Core transformer block design.

- `Classic GPT`: uses learned positional embeddings, LayerNorm, and GELU MLP.
  This is the original Micro LLM Creator architecture and is best for old
  checkpoints.
- `Llama-like`: uses RoPE positional encoding, RMSNorm, and SwiGLU MLP. This is
  closer to modern Llama-style model blocks and is the better default for new
  serious experiments.

Effect:

- `Classic GPT` is simple and stable for tiny tests.
- `Llama-like` usually gives better inductive bias for longer context and modern
  decoder-only language modeling.
- Checkpoints are not interchangeable between block styles.

Recommendation:

- Use `Llama-like` for new models.
- Use `Classic GPT` when resuming older checkpoints created before this option
  existed.

### n_embd

Embedding/channel width.

Effect:

- Larger values increase model capacity.
- Larger values increase memory and training time.

Examples:

- `128`: tiny experiments.
- `256`: small model.
- `512`: stronger small model.

### n_head

Number of attention heads.

Effect:

- More heads can learn different token relationships.
- `n_embd` must divide evenly by `n_head`.

Examples:

- `128 / 4`
- `256 / 4`
- `512 / 8`

### Attention

Attention controls how query heads share key/value heads.

- `Multi-head`: classic full multi-head attention.
- `Grouped-query`: shares key/value heads across groups of query heads. This can
  reduce memory and improve generation efficiency.
- `Multi-query`: all query heads share one key/value head. This is very
  memory-efficient but changes model behavior.

`KV heads` controls grouped-query key/value head count. It must divide `n_head`.
It is ignored by normal multi-head attention and forced to one for multi-query
attention.

`Backend` controls the attention kernel:

- `SDPA / Flash when available`: lets PyTorch use its scaled-dot-product
  attention path. On supported CUDA systems, PyTorch may use Flash Attention
  internally.
- `Manual`: uses the app's explicit attention implementation. It is useful for
  debugging but can be slower.

`Window` enables sliding-window attention. `0` means full context. A positive
number restricts attention to recent tokens.

### n_layer

Number of transformer blocks.

Effect:

- More layers increase depth and pattern capacity.
- More layers train slower.

Examples:

- `4`: tiny.
- `6`: small.
- `8`: stronger small model.

### Context Length

Training context length.

Effect:

- Must match or be less than prepared context intent.
- Longer values use more memory.

For code:

- `256` is a practical minimum for useful snippets.
- `512+` is better if hardware allows.

### Dropout

Regularization rate.

Effect:

- Helps reduce overfitting.
- Too much dropout can weaken learning.

Recommendation:

- `0.1` default.
- `0.0` for very small experiments.
- `0.1` to `0.2` when overfitting.

### Epochs

Number of full passes over the training data.

Effect:

- More epochs can improve learning.
- Too many epochs overfit small datasets.

Recommendation:

- Start with `1` for smoke tests.
- Use `5` to `20` for small experiments.
- Watch validation loss.

### Batch Size

Number of sequences per batch.

Effect:

- Larger batch is smoother but uses more memory.
- Smaller batch works on weaker hardware.

Recommendation:

- CPU: `1` to `4`
- Low VRAM GPU: `4` to `16`
- More VRAM: `16+`

### Learning Rate

Optimizer step size.

Effect:

- Too high causes unstable loss.
- Too low trains slowly.

Recommendation:

- Start with `0.0003`.
- If loss explodes, try `0.0001`.

### Weight Decay

Regularization applied by AdamW.

Effect:

- Helps prevent overfitting.
- Too high can underfit.

Recommendation:

- `0.1` default.
- `0.01` for smaller datasets if learning seems weak.

### Optimizer

Controls how model weights are updated.

- `AdamW`: safest default for small LLM training.
- `Adam`: classic Adam without decoupled weight decay behavior.
- `Lion`: experimental sign-based optimizer. Can work well, but tune carefully.
- `Adafactor`: memory-conscious optimizer when supported by your PyTorch build.

### Schedule

Controls learning-rate changes over time.

- `Warmup linear`: warm up, then linearly decay.
- `Cosine decay`: warm up, then smoothly decay. Good general default.
- `Polynomial decay`: warm up, then polynomial decay controlled by `Poly power`.
- `One-cycle`: rises and falls over the run. Useful for experiments.
- `Constant`: warm up, then keep LR steady.

`Min LR` controls the lowest schedule multiplier. `Poly power` controls the
shape of polynomial decay.

### Gradient Accumulation

Accumulates gradients across multiple batches before updating.

Effect:

- Simulates larger batch sizes without extra memory.
- Slower per optimizer step.

Example:

- Batch size `4`, accumulation `8` behaves like effective batch `32`.

### Warmup Steps

Steps used to ramp up learning rate.

Effect:

- Stabilizes early training.
- Too many warmup steps can delay learning.

Recommendation:

- `100` for small runs.
- `1000+` for larger runs.

### Eval Interval

Steps between validation checks.

Effect:

- More frequent validation gives better visibility.
- Validation pauses training briefly.

Use `0` to skip interval validation.

### Save Interval

Steps between checkpoints.

Effect:

- Lower interval improves crash recovery.
- More checkpoints use more disk.

Recommendation:

- `500` default.
- Lower it for unstable hardware or long runs.

### Max Grad Norm

Gradient clipping limit.

Effect:

- Prevents exploding gradients.
- Too low can slow learning.

Recommendation:

- `1.0` default.

### Seed

Random seed.

Effect:

- Makes initialization and data order more repeatable.

### Device

Training hardware.

- `cuda`: NVIDIA GPU.
- `cpu`: CPU fallback.

Effect:

- CUDA is much faster.
- CPU is useful for smoke tests.

### Use Mixed Precision on CUDA

Uses AMP mixed precision.

Effect:

- Usually faster on NVIDIA GPUs.
- Reduces VRAM usage.

Recommendation:

- Keep enabled on CUDA.

### Precision

Numeric precision for mixed precision training.

- `FP16`: fast and memory-efficient on many NVIDIA GPUs.
- `BF16`: more numerically stable on GPUs that support BF16.
- `FP32`: safest but uses more memory and is usually slower.

### Resume from Latest Checkpoint

Continues interrupted training.

Effect:

- Loads model, optimizer, scheduler, scaler, epoch, and step state.
- Prevents losing long training runs.
- Also supports continued training after adding more data, as long as the
  tokenizer and architecture stay compatible.

Important:

- Keep the same model output folder to continue the same model.
- Keep the same tokenizer when adding more data.
- Keep `n_embd`, `n_head`, `n_layer`, context length, and bias compatible with
  the checkpoint.

### Require Compatible Resume

Validates continued training before loading a checkpoint.

Effect:

- Compares the dataset tokenizer with the tokenizer saved in the model folder.
- Compares checkpoint architecture with the selected UI architecture.
- Stops early with a clear message if the run would be incompatible.

Recommendation:

- Keep enabled for professional work.
- Disable only when debugging old checkpoints manually.

### Resume Checkpoint

Optional exact checkpoint file.

Effect:

- Use this when you want to resume a specific checkpoint instead of the latest.

### Benchmark Prompts

Fixed prompts used to test a trained checkpoint from the `Bench` tab.

Effect:

- Runs the same prompts against `final_model.pt`.
- Saves outputs to `benchmarks/benchmark_<timestamp>.json` inside the model
  folder.
- Records generated token count and generated token speed.
- Helps compare model versions beyond train/validation loss.

Recommendation:

- Keep a small stable set of prompts for every project.
- Include prompts for explanation, code writing, debugging, and code review.
- Compare benchmark outputs after each dataset version or training run.

### Use KV Cache

Reuses attention key/value tensors while generating benchmark answers from a
MicroGPT checkpoint.

Effect:

- Speeds up autoregressive generation because the model does not recompute the
  whole prompt for every new token.
- Is used only for inference/benchmark generation, not training.
- Benchmark JSON records whether KV cache was enabled.

Recommendation:

- Keep enabled for normal benchmark runs.
- Disable only when debugging generation differences.

Example prompts:

````text
Explain what a Python function is and give a tiny example.

Write a Python function that adds two numbers.

Review this code and explain any issue:
```python
def add(a, b):
print(a + b)
```
````

## 4.1 Training Metrics and Telemetry

The `AI` tab shows live training telemetry while a run is active.

Training graphs are interactive. They show titles, legends, X/Y axes, grid
lines, and hover values for nearby plotted points. The X-axis is the optimizer
step. The Y-axis depends on the graph: loss, optimization value, parameter
stability value, throughput rate, or GPU memory in GB.

### ETA

Estimated time remaining based on recent completed training steps.

Effect:

- Gives a practical time estimate after enough steps have completed.
- May fluctuate early in training while speed stabilizes.
- Changes when validation, checkpoint saving, or hardware load affects speed.

### Epoch and Step

Current epoch and optimizer step progress.

Effect:

- Shows how far the run has progressed.
- Helps confirm resume behavior after interruption.

### Train Loss

Current training loss.

Effect:

- Measures how well the model fits the training tokens.
- Should usually decrease over time.

### Validation Loss

Loss on held-out validation tokens.

Effect:

- Measures generalization.
- If validation loss rises while training loss falls, the model may be overfitting.

### Learning Rate

Current learning rate from the scheduler.

Effect:

- Helps diagnose warmup and decay behavior.
- Loss spikes can sometimes correlate with too much learning rate.

### Gradient Norm

Magnitude of gradients before/after clipping.

Effect:

- Large spikes can indicate unstable training.
- Values collapsing toward zero can indicate stalled learning.

### Weight Norm

Magnitude of model parameters.

Effect:

- Helps monitor parameter stability during longer runs.

### Parameter Update Ratio

Approximate size of updates relative to parameter size.

Effect:

- Very large values can destabilize training.
- Very tiny values can mean the model is barely learning.

### Tokens/sec and Samples/sec

Training throughput.

Effect:

- Shows hardware and data pipeline speed.
- Useful when tuning batch size, context length, and GPU settings.

### VRAM Usage

CUDA memory allocated/reserved during GPU training.

Effect:

- Helps identify memory bottlenecks.
- Useful when choosing batch size, context length, and model size.

## 5. Export Options

### Model Core

Folder containing the trained model.

Must contain:

- `final_model.pt`
- `tokenizer.json`
- `training_summary.json`

Optional but recommended:

- `model_lineage.json`
- `dataset_summary.json`
- `benchmarks/`

### Output Bay

Destination folder for exports.

### Quantization

Available now:

- `FP16 checkpoint`

Requires a real llama.cpp-compatible HF model or a custom MicroGPT converter:

- `GGUF Q8_0`
- `GGUF Q4_K_M`
- `GGUF Q5_K_M`

FP16 effect:

- Smaller checkpoint.
- Useful for inference/conversion workflows.

GGUF note:

- GGUF export should be done through a valid llama.cpp/Hugging Face-compatible
  conversion path. The app intentionally avoids writing fake GGUF files.
- `Convert HF to GGUF` runs llama.cpp's `convert_hf_to_gguf.py` when the model
  core contains a real `hf_model` folder.
- Native MicroGPT checkpoints are not directly GGUF-compatible yet.
- `Export HF Package` creates a Hugging Face-style MicroGPT folder, but it uses
  `model_type: microgpt`, which llama.cpp does not support unless a custom
  converter/model implementation is added.

### llama.cpp

Path to a local llama.cpp checkout containing:

```text
convert_hf_to_gguf.py
```

### GGUF Output

Destination `.gguf` file path.

### GGUF Outtype

Output type passed to llama.cpp conversion.

- `f16`: recommended starting point.
- `f32`: larger, mostly useful for debugging.
- `bf16`: useful on hardware/workflows that prefer bfloat16.
- `q8_0`: supported by the llama.cpp converter for compatible HF models.
- `q8_0`: converter-supported quantized output when available.

### Create Bundle

Copies model artifacts into an export folder.

The bundle includes required model files plus lineage and benchmark artifacts
when available.

### Quantize Model

Creates an FP16 checkpoint today.

### Export HF Package

Creates:

```text
model_core/hf_model/
```

The folder contains:

- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `generation_config.json`
- `training_summary.json`
- `model_lineage.json` when available
- `dataset_summary.json` when available
- `README.md`

This is useful for portability and future converter work. It is not a claim
that the model is a Llama-compatible Hugging Face model.

### Convert HF to GGUF

Runs llama.cpp conversion for:

```text
model_core/hf_model
```

Use this only when `hf_model` is a real Hugging Face-compatible model folder.
The app will fail with a clear message instead of writing a fake GGUF file.

## 6. Benchmark Tab

The `Bench` tab runs fixed prompts against a trained MicroGPT checkpoint. Use it
to compare model versions after changing data, tokenizer policy, architecture,
or training settings.

The benchmark panel includes:

- Prompt list separated by blank lines.
- Max tokens.
- Temperature.
- KV cache toggle.
- Run and stop controls.
- Benchmark telemetry log.

Outputs are saved under the model folder in `benchmarks/`.

## 7. Test Chat Options

The `Chat` tab is for trying a GGUF model through llama.cpp without reloading it
for every message.

Replies stream into the chat window and are rendered as Markdown. Fenced code
blocks are syntax-highlighted when the `markdown` and `Pygments` packages from
`requirements.txt` are installed.

### GGUF Model

Path to the `.gguf` file to load.

Effect:

- The model is loaded once in the background.
- Later chat messages reuse the loaded model.
- Large GGUF files can take time and memory to load.

### Context

The llama.cpp context window.

Effect:

- Larger context supports longer conversations.
- Larger context uses more memory.

### CPU Threads

Number of CPU threads used for inference.

Effect:

- Higher values can improve speed.
- Too high can make the desktop less responsive.

### GPU Layers

Number of model layers to offload to GPU when supported.

Effect:

- More GPU layers can improve speed.
- Requires a compatible llama.cpp build and enough VRAM.
- `-1` asks llama.cpp to offload all possible layers.

Recommendation:

- Use `-1` when you want the model to load on GPU.
- If loading fails, install a GPU-enabled `llama-cpp-python` build or reduce the layer count.
- If GPU layers are not `0` and the installed llama runtime is CPU-only, the app stops loading and shows a clear error instead of silently using CPU.

Recommended CUDA install example using a prebuilt wheel:

```powershell
pip uninstall -y llama-cpp-python
pip install --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Use the wheel folder that matches your CUDA version, such as `cu121`, `cu122`,
`cu123`, `cu124`, `cu125`, `cu130`, or `cu132`.

Source-build CUDA install example:

```powershell
pip uninstall -y llama-cpp-python
$env:CMAKE_ARGS="-DGGML_CUDA=on"
$env:FORCE_CMAKE="1"
pip install --no-cache-dir --force-reinstall llama-cpp-python
```

If source build fails with `CUDA Toolkit not found` or `Could not find nvcc`,
install NVIDIA CUDA Toolkit first and ensure `nvcc --version` works in the same
terminal.

### Thinking

Turns reasoning-style prompting on or off.

Effect:

- When enabled, the app adds an instruction style based on Reasoning Effort.
- When disabled, the app asks for a more direct answer.
- This changes prompting behavior; it does not retrain the model.

Recommendation:

- Keep enabled when testing explanation, debugging, or review behavior.
- Turn off when you want short direct answers.

### Reasoning Effort

Instruction style sent with each prompt.

- `Fast`: shorter, speed-focused replies.
- `Balanced`: clear default behavior.
- `Deep`: asks the model for more careful reasoning.

### Max Tokens

Maximum new tokens for each reply.

Effect:

- Higher values allow longer answers.
- Higher values take longer to generate.

### Temperature

Sampling randomness.

Effect:

- Lower values are more focused.
- Higher values are more creative but less predictable.

### Top-p

Nucleus sampling cutoff.

Effect:

- Lower values restrict output to more likely tokens.
- Higher values allow more variety.

### Repeat Penalty

Penalty for repeated text.

Effect:

- Higher values can reduce loops.
- Too high can make wording unnatural.

### System Prompt

Optional behavior instruction for the chat.

Effect:

- Helps steer style, role, and answer format.
- Does not retrain the model.

## 8. Suggested Settings

### Smoke Test

Use this to verify the pipeline works.

- Context: `64` or `128`
- n_embd: `32` to `128`
- n_head: `4`
- n_layer: `2` to `4`
- Epochs: `1`
- Batch size: `1` to `4`
- Device: `cpu`

### Small Code Model

Use this for a first real code experiment.

- Code Training Mode: enabled
- Include source files: enabled
- Extract code blocks: enabled
- Preserve indentation: enabled
- Context: `256` or `512`
- n_embd: `256`
- n_head: `4`
- n_layer: `6`
- Batch size: as high as your GPU allows
- Learning rate: `0.0003`
- Epochs: `5` to `20`

### Stronger Small Model

Use this when you have more data and VRAM.

- Context: `512` to `1024`
- n_embd: `512`
- n_head: `8`
- n_layer: `8`
- Batch size: `8+`
- Gradient accumulation: increase if VRAM is limited

## 9. Programming PDFs: Best Practice

Programming PDFs help most when used as explanation data. Raw PDF code is often
damaged during extraction. For best results:

1. Keep Code Training Mode enabled.
2. Keep explanations enabled.
3. Add real source-code folders when possible.
4. Preserve indentation.
5. Inspect `corpus.txt` after preparation.
6. Remove PDFs that extract badly.

Good training mix:

- Books/tutorial explanations.
- Real source files.
- README files.
- Tests.
- Small examples.
- Q&A/instruction style data.

Avoid:

- OCR-damaged PDFs.
- Minified code.
- Huge generated files.
- Vendor/build folders.
- Duplicate content.

## 10. How to Know Training Is Working

Good signs:

- Training loss decreases.
- Validation loss decreases or stabilizes.
- ETA and step counters continue moving.
- Tokens/sec and samples/sec stay reasonably stable.
- Generated samples become more structured.
- Code indentation improves.

Bad signs:

- Loss becomes `nan`.
- Validation loss rises while training loss falls.
- Gradient norm spikes repeatedly.
- VRAM usage approaches the hardware limit.
- Generated text repeats endlessly.
- Code loses indentation.

Fixes:

- Lower learning rate.
- Add more clean data.
- Reduce model size for small datasets.
- Increase validation split slightly.
- Use source-code files instead of PDF-only code.

## 11. Important Limitations

This app trains small models from scratch. A small model will not automatically
match large commercial coding models. To improve behavior, you need:

- Clean data.
- Enough tokens.
- Good tokenizer settings.
- Reasonable model size.
- Instruction-style examples.
- Reasoning-shaped examples.
- Evaluation prompts.

For "thinking" behavior, train on examples that show real problem-solving,
debugging, explanation, and code review patterns. The app can scaffold the
format, but the quality comes from the data.
