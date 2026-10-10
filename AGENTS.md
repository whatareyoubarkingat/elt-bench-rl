# Project Language

All project artifacts must use English, including documentation, reports,
code comments, docstrings, tool prompts, error messages, configuration notes,
filenames, and commit messages. Do not add Chinese characters to these artifacts.

Keep third-party source checkouts in the external pinned upstream cache rather
than vendoring multilingual examples into this repository. Preserve the recorded
upstream revisions and verifier integrity. Secrets, downloaded tokenizer assets,
and execution outputs remain local and untracked.

Preserve honest validation boundaries: offline tests and scripted pipelines do
not establish a real model rollout or a real Tinker optimization step.
