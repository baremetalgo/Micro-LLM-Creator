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

The app has three main work areas:

- `IN`: prepare datasets.
- `AI`: configure and train the model.
- `X`: export and quantize model artifacts.

## 2. Recommended Workflow

1. Collect clean training material.
2. Open `IN`.
3. Choose the source folder.
4. Enable Code Training Mode if your data contains programming material.
5. Prepare the dataset.
6. Open `AI`.
7. Choose a model size and training settings.
8. Start training.
9. Resume training if interrupted.
10. Open `X`.
11. Bundle or quantize the trained model.

## 3. Dataset Preparation

Dataset preparation converts your files into a tokenizer-ready corpus. It also
trains a byte-level BPE tokenizer and splits tokens into training and validation
streams.

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

### Dataset Core

The output folder for prepared artifacts.

The app writes:

- `corpus.txt`
- `tokenizer.json`
- `train_tokens.json`
- `val_tokens.json`
- `dataset_summary.json`

Effect on the LLM:

- This folder becomes the training source for the `AI` tab.
- Reusing the same dataset keeps experiments comparable.

### Parallel Lanes

Number of files read in parallel.

Effect:

- Higher values can speed up hundreds of PDFs/text files.
- Very high values can make disk usage and CPU load heavy.
- A good starting value is `4` to `8`.

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
- Useful for “explain this code” behavior.
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

```text
<sample type="code" language="python" source="example.py">
<instruction>Study this python code and learn its syntax, structure, and patterns.</instruction>
<code>
def hello():
    return "hi"
</code>
</sample>
```

Effect:

- Gives the model a hint that code is a task-oriented sample.
- Useful for instruction-style prompting later.

Recommendation:

- Keep enabled for general coding assistant behavior.

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
- `checkpoints/`

### Preset

Quick architecture presets.

- `Tiny`: faster, lower quality, good for testing.
- `Small`: more capacity, needs more data and memory.
- `Custom`: use your own values.

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

### Resume from Latest Checkpoint

Continues interrupted training.

Effect:

- Loads model, optimizer, scheduler, scaler, epoch, and step state.
- Prevents losing long training runs.

### Resume Checkpoint

Optional exact checkpoint file.

Effect:

- Use this when you want to resume a specific checkpoint instead of the latest.

## 5. Export Options

### Model Core

Folder containing the trained model.

Must contain:

- `final_model.pt`
- `tokenizer.json`
- `training_summary.json`

### Output Bay

Destination folder for exports.

### Quantization

Available now:

- `FP16 checkpoint`

Planned:

- `GGUF Q8_0`
- `GGUF Q4_K_M`
- `GGUF Q5_K_M`

FP16 effect:

- Smaller checkpoint.
- Useful for inference/conversion workflows.

GGUF note:

- GGUF export should be done through a valid llama.cpp/Hugging Face-compatible
  conversion path. The app intentionally avoids writing fake GGUF files.

### Create Bundle

Copies model artifacts into an export folder.

### Quantize Model

Creates an FP16 checkpoint today.

## 6. Suggested Settings

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

## 7. Programming PDFs: Best Practice

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

## 8. How to Know Training Is Working

Good signs:

- Training loss decreases.
- Validation loss decreases or stabilizes.
- Generated samples become more structured.
- Code indentation improves.

Bad signs:

- Loss becomes `nan`.
- Validation loss rises while training loss falls.
- Generated text repeats endlessly.
- Code loses indentation.

Fixes:

- Lower learning rate.
- Add more clean data.
- Reduce model size for small datasets.
- Increase validation split slightly.
- Use source-code files instead of PDF-only code.

## 9. Important Limitations

This app trains small models from scratch. A small model will not automatically
match large commercial coding models. To improve behavior, you need:

- Clean data.
- Enough tokens.
- Good tokenizer settings.
- Reasonable model size.
- Instruction-style examples.
- Evaluation prompts.

For “thinking” behavior, train on examples that show step-by-step reasoning,
debugging, explanation, and code review patterns.

