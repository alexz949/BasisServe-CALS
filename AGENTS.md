# Agent Instructions
# Writing Codes
- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. 
- Never trade a working product for unfinished complexity. Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- DO NOT write any raise error or execption codes.
## Experiments, Training, Evaluation, and Analysis

- Use the `lowrank` or `basis` conda environment for all experiments, training, evaluation, and analysis scripts.

- Run substantial jobs through Slurm. Do not launch long-running experiments directly on the login node.
- You can check all possible slurm partition through commands
- partition `yangGrp` is our node, but only use it when all other nodes are full.
When preparing Slurm jobs:

- Try to request as many GPUs as are reasonably useful and available for the task.
- Avoid over-requesting CPUs. Use the minimum number of CPUs needed to support the requested GPUs and dataloading.
- Prefer GPU-heavy, CPU-light allocations unless the task clearly requires more CPU resources.
- Output folder name and file name should be light but clear.
Before running any job, evaluation, or large script:

- Summarize the exact command you are about to run, not Slurm command.
- Every time when create and submit a sbatch file, remember to delete.
- Ask for explicit confirmation before launching the job.
- When running the job, make sure to include some log files so either you and me can trace the progress.
- Avoid creating too many retry-ish folder, if a job fails, resubmit with the same settings
If the user's instruction is ambiguous, incomplete, or underspecified:

- Do not guess silently.
- Ask a clarifying question before running code or changing important files.

When reporting results, include:

- The command used.
- The conda environment.
- Key metrics and any obvious warnings or failures.

This is a GitHub repository belongs to https://github.com/alexz949/BasisServe-CALS


## Repository Context

- Treat all uploaded files as repository inputs unless the user explicitly says otherwise.
- Treat generated summary Markdown files (`*.md`) as repository artifacts that may need to be uploaded/committed.
- Preserve the original meaning, filenames, and structure of uploaded files whenever possible.
- Do not delete, rename, overwrite, or reorganize files unless the user explicitly asks.

## GitHub Upload / Commit Workflow

After every run that creates, edits, or summarizes files, ask the user:

1. Whether they want the uploaded files and generated summary Markdown files uploaded to the GitHub repository.
2. What commit message they want to use.

Do **not** upload, commit, push, or otherwise modify the GitHub repository until the user explicitly confirms.


## Commit Message Handling

- If the user provides a commit message, use it exactly unless it is clearly unsafe or malformed.
- If the user asks for a suggested commit message, propose one concise message.
- If multiple unrelated changes were made, suggest separating them into multiple commits.
- Never invent a commit message and commit automatically.

## Safety and Confirmation

Before any GitHub write action:

- Confirm the exact files to upload.
- Confirm the exact commit message.
- Confirm whether to push to the remote branch, if applicable.

If GitHub access is unavailable, provide the user with the exact `git` commands they can run locally.