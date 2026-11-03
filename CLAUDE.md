GLOBAL RULE: STOP OVERENGINEERING. NO CEREMONY. NO REHEARSALS.

1. Build the smallest working solution that fully satisfies the request. Reuse existing code, tools, tests, and standard-library features. Do not invent requirements, speculative abstractions, or future-proofing.

2. Do the actual work. Do not create unnecessary planning rounds, wrappers, validators, manifests, reports, evidence pipelines, or duplicate implementations. Supporting work must remain smaller than the task it supports.

3. No rehearsals. Exercise the real changed flow using existing checks. Use diagnostics only to answer a specific question blocking progress.

4. Stop auditing the audit. Review the actual implementation, not reviews, handoffs, logs, or verification machinery. Default to one independent review and one focused re-review of concrete fixes. Additional rounds require a concrete new failure, relevant code change, or explicit user requirement. Do not repeat passing checks on unchanged code.

5. Preserve correctness, security, data integrity, accessibility, required tests, real-app verification, and explicit approval gates. Do not invent additional gates. When preparation starts expanding recursively, stop expanding it, fix the concrete blocker, and continue delivery.

Done means working, verified functionality delivered—not more preparation artifacts.

IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
