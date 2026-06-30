# How To Create Your LLM

## Introduction

Micro LLM Creator is a desktop workflow for building a small local language
model from your own text, PDFs, and source-code files. The app guides you
through four main stages:

1. Prepare a clean dataset.
2. Choose tokenizer, model, and training settings.
3. Train, resume, or fine-tune the model.
4. Benchmark, export, and test the model in chat.

The most important rule is consistency: once you train a model family, keep the
same tokenizer and compatible architecture when you continue training or
fine-tune from an existing checkpoint.

## Workflow Overview

Use staged training:

1. **Base pretraining**: train from scratch on local text/code/PDF material and
   optional base corpora. This teaches grammar, language patterns, code syntax,
   and next-token prediction.
2. **Instruction fine-tuning**: load the base checkpoint and train on
   instruction data such as Alpaca, Dolly, or SlimOrca. This teaches
   request-following.
3. **Conversation fine-tuning**: load the base or instruction checkpoint and
   train on chat data such as UltraChat, DailyDialog, or OpenAssistant. This
   teaches assistant-style conversation.
4. **Export and test**: export/convert the model, then test it in Chat.

TinyStories is optional base-pretraining data only. It is not selected by
default and is not used for instruction or conversation fine-tuning.

## Step 1: Create Or Open A Project

1. Type a project name in the top bar.
2. Click `New Project` if you want a clean project workspace.
3. Click `Save Project` to create project folders.
4. Use `Open Project` later to restore the saved paths and settings.

The app creates a standard workspace:

```text
project/
  project.json
  data/
  datasets/
  models/
  exports/
  cache/
  temp/
```

### Project Parameters

`Project name`

- Human-readable project name.
- Used when creating the project folder.

`New Project`

- Clears current settings and starts a new project configuration.

`Save Project`

- Saves paths, dataset settings, training settings, export settings, chat
  settings, and artifact summaries to `project.json`.
- If the project was already saved, it saves over the existing project file.

`Open Project`

- Loads an existing `project.json`.
- Restores paths and UI choices.

## Step 2: Prepare Your Dataset

Open the `IN` tab.

The dataset stage reads your files, extracts text/code, creates or reuses a
tokenizer, tokenizes the corpus, and writes training/validation token files.

Recommended first run:

1. Choose your source folder.
2. Choose or keep the dataset output folder.
3. Set `Dataset purpose` to `Base pretraining`.
4. Leave `Include online training datasets` off unless you intentionally want
   an online base corpus.
5. Click `Preview Dataset`.
6. Review duplicate and extraction warnings.
7. Enable `Code Training Mode` if your data contains programming content.
8. Keep tokenizer strategy as `Auto` for a new model.
9. Click `Prepare Dataset`.

### Source Array Parameters

`Source vault`

- Folder containing input files.
- Can include PDFs, text files, markdown, JSONL, and source-code files.

Effect:

- More clean data usually improves model behavior.
- Noisy PDFs can teach broken text, bad spacing, or corrupt code.

`Dataset core`

- Output folder for prepared dataset artifacts.
- Stores `corpus.txt`, `tokenizer.json`, `train_tokens.json`,
  `val_tokens.json`, cache files, summaries, and lineage.

`Parallel lanes`

- Number of worker lanes used during dataset preparation.

Effect:

- Higher values can process many files faster.
- Too high can use more memory and disk activity.

Recommendation:

- Use 4 to 8 for normal machines.
- Use more only if you have many CPU cores and enough memory.

`Prepare mode`

- `Incremental update`: reuse cached extraction for unchanged files.
- `Full rebuild`: rebuild the dataset from known source files.
- `Force reprocess`: ignore extraction cache and re-read everything.

Recommendation:

- Use `Incremental update` when adding more files later.
- Use `Force reprocess` after changing extraction/code options.

`Lowercase text`

- Converts text to lowercase before training.

Effect:

- Reduces vocabulary variety.
- Loses case-sensitive information, which is bad for code.

Recommendation:

- Keep off for programming data.

`Code Training Mode`

- Enables code-aware processing.
- Adds code/prose handling and code-specific extraction options.

Recommendation:

- Enable for programming PDFs or source-code folders.

`Include source files`

- Includes actual source-code files such as `.py`, `.js`, `.cpp`, `.java`,
  `.cs`, and others.

Recommendation:

- Enable when training a code-capable model.

`Dataset purpose`

- `Base pretraining`: creates a dataset for training from scratch.
- `Instruction fine-tune`: creates a dataset for request-following adaptation.
- `Conversation fine-tune`: creates a dataset for chat/dialogue adaptation.

Effect:

- Controls which online datasets are visible.
- Helps keep base training separate from later fine-tuning.

Recommendation:

- Use `Base pretraining` for your first model.
- Use `Instruction fine-tune` only after you have a compatible base checkpoint.
- Use `Conversation fine-tune` after the model has learned language basics.

`Include online training datasets`

- Enables selected Hugging Face datasets for the current dataset purpose.
- Disabled by default.
- No dataset is selected by default.

Base pretraining options:

- `TinyStories`: simple stories for basic fluency.
- `WikiText-103`: clean Wikipedia-style long-form prose.
- `Wikipedia EN 2023`: broad encyclopedia prose.
- `FineWeb-Edu sample`: educational web text.

Instruction fine-tune options:

- `Alpaca 52K`: instruction-following pairs.
- `Dolly 15K`: human-written instruction examples.
- `SlimOrca`: instruction and reasoning-style answers.

Conversation fine-tune options:

- `UltraChat 200K`: assistant conversations.
- `DailyDialog`: natural daily dialogue.
- `OpenAssistant OASST1`: assistant-style conversation data.

`Rows per dataset`

- Maximum rows read from each selected online dataset.
- Use a small number for testing.
- Use larger values only when disk, network, and training time are available.

## Step 3: Configure Tokenizer And Dataset Quality

### Tokenizer Core Parameters

`Auto vocabulary`

- Lets the app choose vocabulary size after reading the dataset.

Effect:

- Small datasets get smaller vocabularies.
- Larger datasets can use larger vocabularies.

Recommendation:

- Keep enabled for most projects.

`Manual vocabulary`

- Vocabulary size used when auto vocabulary is disabled.

Effect:

- Larger vocabulary can preserve more words and code symbols.
- Larger vocabulary increases model output layer size.

Common values:

- `4000`: tiny experiments.
- `8000`: small mixed text/code datasets.
- `16000`: larger datasets.
- `32000`: larger serious datasets.

`Selected vocab`

- Displays the vocabulary size the app selected or calculated.

`Tokenizer policy`

- `Auto`: choose the safest available behavior.
- `Train new tokenizer`: build a new tokenizer from current corpus.
- `Reuse dataset tokenizer`: use existing `tokenizer.json` in the dataset
  folder.
- `Import tokenizer.json`: copy tokenizer from another compatible project.

Recommendation:

- Use `Train new tokenizer` for a brand-new model family.
- Use `Reuse dataset tokenizer` or `Import tokenizer.json` when adding more
  data to an existing model family.

`Import tokenizer`

- Path to tokenizer JSON used by `Import tokenizer.json`.

Effect:

- Keeps token IDs stable across model versions.
- Required for safe continued training across related datasets.

`Min frequency`

- Minimum frequency for a token candidate to enter the tokenizer vocabulary.

Effect:

- Higher values remove rare fragments.
- Lower values keep more rare words/symbols.

Recommendation:

- Use `2` for most datasets.

`Context window`

- Token window used during dataset preparation.

Effect:

- Should match or support your intended training context length.
- Longer context gives the model more surrounding information.

Recommendation:

- Use `256` minimum for code.
- Use `512` or `1024` if memory allows.

`Validation split`

- Fraction of tokens held out for validation.

Effect:

- Validation loss measures generalization.
- Too much validation reduces training data.

Recommendation:

- Use `0.1` for most datasets.
- Use `0.05` for very small datasets.

`Include explanations`

- Keeps prose explanations when code mode is active.

Effect:

- Helps with “explain this code” and assistant-style answers.

`Extract code blocks`

- Attempts to identify code-like blocks inside PDFs/text files.

Effect:

- Better preserves programming examples copied from books.

`Preserve indentation`

- Keeps indentation and line structure.

Recommendation:

- Keep enabled for code training.

`Instruction-style samples`

- Wraps some code/prose examples into task-like samples.

Effect:

- Helps the model learn prompt/answer style behavior.

`Reasoning samples`

- Controls whether examples include reasoning scaffolds.

Options:

- `Reasoning scaffold`: compact reasoning hints.
- `Detailed code reasoning`: more detailed reasoning examples.
- `No reasoning wrapper`: plain text/code samples.

Recommendation:

- Use `Reasoning scaffold` for a balanced small model.

### Dataset Quality Parameters

`Samples`

- Number of extracted training samples.

`Tokens`

- Number of encoded tokens.

Effect:

- More tokens usually improves training.
- Very small token counts produce smoke-test models, not capable assistants.

`Vocab`

- Final tokenizer vocabulary size.

`Code/prose`

- Balance between code samples and explanatory text.

`Cache`

- Number of files reused from extraction cache.

`Warnings`

- Dataset quality warnings such as low token count, duplicate files, or bad
  PDF extraction.

`Ingest telemetry`

- Log of preparation progress, skipped files, processed files, and warnings.

## Step 4: Choose Model Architecture

Open the `AI` tab.

Recommended beginner settings for a first useful small model:

- `Preset`: Small
- `Block style`: Llama-like
- `n_embd`: 512
- `n_head`: 8
- `n_layer`: 8
- `Context length`: 512 or 1024
- `Attention`: Multi-head or Grouped-query
- `Backend`: SDPA / Flash when available

### Model Architecture Parameters

`Dataset`

- Prepared dataset folder.
- Must contain `tokenizer.json`, `train_tokens.json`, and `val_tokens.json`.

`Model`

- Output folder for checkpoints, summaries, lineage, and final model files.

`Preset`

- `Tiny`: quick tests.
- `Small`: stronger small model.
- `Custom`: keep your manual architecture values.

`Block style`

- `Classic GPT`: learned positions, LayerNorm, GELU.
- `Llama-like`: RoPE, RMSNorm, SwiGLU.

Recommendation:

- Use `Llama-like` for new serious experiments.
- Use `Classic GPT` only for older compatible checkpoints.

`n_embd`

- Embedding/channel width.

Effect:

- Higher values increase model capacity.
- Higher values use more memory and train slower.

Examples:

- `128`: tiny.
- `256`: small.
- `512`: stronger small model.

`n_head`

- Number of attention heads.

Rule:

- `n_embd` must divide evenly by `n_head`.

Examples:

- `128 / 4`
- `256 / 4`
- `512 / 8`

`Attention`

- `Multi-head`: standard full attention.
- `Grouped-query`: shares key/value heads across groups.
- `Multi-query`: all query heads share one key/value head.

Effect:

- Grouped-query and multi-query can reduce memory and generation cost.
- They change checkpoint shape, so they are not interchangeable with normal
  multi-head checkpoints.

`KV heads`

- Key/value head count for grouped-query attention.

Rule:

- Must divide `n_head`.

`Backend`

- `SDPA / Flash when available`: lets PyTorch use optimized attention kernels.
- `Manual`: explicit attention implementation, useful for debugging.

Recommendation:

- Use `SDPA / Flash when available`.

`Window`

- Sliding attention window.

Effect:

- `0` means full context.
- Positive values restrict attention to recent tokens.

`n_layer`

- Number of transformer blocks.

Effect:

- More layers increase depth and capability.
- More layers train slower and use more memory.

`Context length`

- Training sequence length in tokens.

Effect:

- Longer context helps code and long explanations.
- Longer context uses more memory.

Recommendation:

- `512` for useful code experiments.
- `1024+` if GPU memory allows.

`Dropout`

- Regularization probability.

Effect:

- Helps reduce overfitting.
- Too high can weaken learning.

Recommendation:

- `0.1` default.
- `0.05` for fine-tuning.

## Step 5: Choose Optimization Settings

### Optimization Engine Parameters

`Epochs`

- Number of full passes over the training tokens.

Recommendation:

- Start with `1` for smoke tests.
- Use `5` to `20` for small experiments.

`Batch`

- Number of sequences per training batch.

Effect:

- Larger batches are smoother but use more memory.

`Profile`

- Quick training recipe.

Options:

- `Stable LLM`: AdamW, cosine schedule, balanced defaults.
- `Low-memory`: Adafactor and grouped-query attention.
- `Code fine-tune`: LoRA-friendly code adaptation settings.
- `Experimental Lion`: Lion optimizer with one-cycle schedule.

`Apply Profile`

- Applies selected profile values to the visible controls.

`LR`

- Learning rate.

Effect:

- Too high destabilizes loss.
- Too low trains slowly.

Recommendation:

- `0.0003` for pretraining.
- `0.00005` to `0.0001` for fine-tuning.

`Decay`

- Weight decay regularization.

Effect:

- Helps reduce overfitting.

`Optimizer`

- `AdamW`: safest default.
- `Adam`: classic Adam.
- `Lion`: experimental sign-based optimizer.
- `Adafactor`: memory-conscious optimizer when supported.

`Schedule`

- `Warmup linear`: warm up, then linear decay.
- `Cosine decay`: smooth decay, strong general default.
- `Polynomial decay`: decay controlled by polynomial power.
- `One-cycle`: learning rate rises and falls.
- `Constant`: warm up, then stay steady.

`Min LR`

- Lowest learning-rate multiplier.

Example:

- `0.1` means decay to 10 percent of base LR.

`Poly power`

- Shape of polynomial decay.

`Grad accum`

- Number of batches to accumulate before optimizer step.

Effect:

- Simulates larger batch size with less memory.

`Warmup`

- Steps used to ramp up learning rate.

`Eval every`

- Steps between validation checks.

`Eval batches`

- Maximum validation batches per validation check.

`Save every`

- Steps between checkpoints.

Effect:

- Smaller values improve crash recovery.
- Smaller values use more disk.

`CPU workers`

- CPU loader workers used to prepare batches.

Effect:

- Helps GPU stay busy.
- Too many workers can increase memory use.

`Max grad`

- Gradient clipping norm.

Effect:

- Helps prevent exploding gradients.

`Seed`

- Random seed for repeatability.

## Step 6: Choose Runtime, Resume, And Fine-Tuning Settings

### Runtime Control Parameters

`Device`

- `cuda`: NVIDIA GPU.
- `cpu`: CPU fallback.

`Hardware`

- Shows detected hardware readiness.

`Mode`

- `Pretrain from scratch`: train from random weights on a base dataset.
- `Instruction fine-tune`: load a compatible checkpoint and train on
  instruction data.
- `Conversation fine-tune`: load a compatible checkpoint and train on chat
  data.
- `Fine-tune checkpoint`: generic checkpoint fine-tuning for custom/domain data.

Fine-tuning workflow:

1. Train a base model first with `Pretrain from scratch`.
2. Keep the base model architecture and tokenizer.
3. Prepare a new instruction or conversation dataset.
4. Use `Reuse dataset tokenizer` or `Import tokenizer.json` from the base model
   family.
5. Select `Instruction fine-tune` or `Conversation fine-tune`.
6. Set `Base model` to the compatible checkpoint.
7. Prefer `LoRA adapters` for efficient fine-tuning.
8. Click `Check Fine-tune`.
9. Start training.

`Base model`

- Base checkpoint path for fine-tuning.
- Must match tokenizer vocabulary and model shape.

`PEFT`

- `Full fine-tune`: update all model weights.
- `LoRA adapters`: freeze base model and train small adapters.

Recommendation:

- Use `LoRA adapters` for most fine-tuning.

`LoRA rank`

- Adapter rank.

Effect:

- Higher rank gives more adapter capacity.
- Higher rank creates larger adapter checkpoints.

Recommendation:

- Start with `8`.
- Try `16` for stronger adaptation.

`LoRA alpha`

- Adapter scaling value.

Recommendation:

- Start with `2 * rank`.

`LoRA dropout`

- Dropout inside LoRA adapters.

Recommendation:

- Use `0.05` for code fine-tuning.

`LoRA target`

- `Attention projections`: attach adapters to attention layers.
- `MLP projections`: attach adapters to feed-forward layers.
- `Attention + MLP`: attach adapters to both.

Recommendation:

- Start with `Attention projections`.
- Use `Attention + MLP` if adaptation is too weak.

`Check Fine-tune`

- Checks whether the selected base checkpoint is compatible.

`Mixed precision`

- Enables CUDA AMP.

`Precision`

- `FP16`: fast and memory-efficient on many GPUs.
- `BF16`: more stable on supported GPUs.
- `FP32`: safest but slower and larger.

`Resume latest`

- Continues the newest checkpoint in the model output folder.

Important:

- Resume means continue an interrupted run.
- Fine-tune means start a fresh optimizer run from base weights.

`Safe resume`

- Requires tokenizer, architecture, optimizer, scheduler, and scaler state to
  match for exact continuation.

`Checkpoint`

- Optional exact checkpoint to resume.

`Check Resume`

- Previews whether resume is safe.

`Resume Compatibility`

- Shows `[OK]`, `[WARN]`, and `[BLOCK]` messages for resume/fine-tune checks.

## Step 7: Start Training

1. Click `Refresh Estimate`.
2. Review estimated model size and VRAM.
3. Click `Check Resume` or `Check Fine-tune` if applicable.
4. Click `Start Training`.
5. Watch the `LIVE` tab for loss, throughput, VRAM, RAM, and progress.
6. Use `Stop` if needed. The app saves a resumable checkpoint.

Training outputs:

```text
models/
  final_model.pt
  tokenizer.json
  training_summary.json
  model_lineage.json
  training_history.json
  checkpoints/
```

LoRA outputs also include:

```text
models/
  final_adapter.pt
  checkpoints/checkpoint_<step>.pt
```

Adapter checkpoints are small. `final_model.pt` is merged so benchmark/export
tools can use it like a normal full checkpoint.

## Step 8: Monitor Training Live

Open the `LIVE` tab.

Important metrics:

`Loss`

- Training loss.
- Should generally decrease.

`Validation loss`

- Measures generalization.
- If training loss drops but validation loss rises, the model is overfitting.

`LR`

- Current learning rate from the scheduler.

`Gradient norm`

- Helps detect unstable gradients.

`Weight norm`

- Tracks parameter magnitude.

`Update ratio`

- Indicates how large updates are compared with model weights.

`Tokens/sec`

- Training throughput.

`Samples/sec`

- Batch throughput.

`GPU memory`

- CUDA memory usage.

`System RAM`

- Host memory usage.

`Timeline slider`

- Lets you inspect saved graph values from telemetry history.

## Step 9: Benchmark The Model

Open the `Bench` tab.

1. Keep or edit benchmark prompts.
2. Choose max generated tokens.
3. Choose temperature.
4. Keep `Use KV cache` enabled.
5. Click `Run Benchmark`.

Benchmark output is saved under:

```text
models/benchmarks/
```

Benchmark JSON records:

- prompt
- output
- elapsed time
- generated token count
- token speed
- model lineage

## Step 10: Export The Model

Open the `X` tab.

Common options:

`Create Bundle`

- Copies model, tokenizer, summaries, and metadata into an export folder.

`Quantize Model`

- Creates smaller lower-precision checkpoint artifacts where supported.

`Export HF Package`

- Writes a Hugging Face-style MicroGPT package for portability.

`Convert HF to GGUF`

- Uses llama.cpp converter when configured.

## Step 11: Test In Chat

Open the `Chat` tab.

1. Choose a GGUF model.
2. Click `Load Model`.
3. Adjust context, GPU layers, temperature, top-p, max tokens, and thinking
   settings.
4. Send prompts.
5. Use `Unload` when done.

Chat is for GGUF inference. Native MicroGPT checkpoints should be benchmarked
from the `Bench` tab or exported first.

## Recommended Starter Recipes

### Tiny Smoke Test

- Dataset: small folder
- Preset: Tiny
- Block style: Classic GPT or Llama-like
- Context length: 128
- Epochs: 1
- Batch: 4 to 16
- Device: CPU or CUDA

Goal:

- Verify end-to-end workflow.

### Small Code Model

- Code Training Mode: enabled
- Tokenizer: Auto
- Preset: Small
- Block style: Llama-like
- n_embd: 512
- n_head: 8
- n_layer: 8
- Context length: 512 or 1024
- Optimizer: AdamW
- Schedule: Cosine decay
- Precision: FP16 or BF16

Goal:

- Train a small coding-focused model.

### LoRA Code Fine-Tune

- Mode: Fine-tune checkpoint
- PEFT: LoRA adapters
- Base model: compatible checkpoint
- Profile: Code fine-tune
- LoRA rank: 8
- LoRA alpha: 16
- LoRA target: Attention projections
- LR: 0.00005
- Dropout: 0.05

Goal:

- Adapt an existing compatible model to new programming material with small
  adapter checkpoints.

## Final Checklist

Before spending hours training:

- Dataset preview is clean.
- Duplicate and bad extraction warnings are reviewed.
- Tokenizer policy is correct.
- Dataset has enough tokens.
- Model size fits hardware.
- CUDA is detected if using GPU.
- Resume or fine-tune compatibility check passes.
- Save interval is reasonable.
- Project is saved.
