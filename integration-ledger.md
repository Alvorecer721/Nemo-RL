# PP foundation integration ledger

User authorized execution with "go" on 2026-09-05. Useful format changes are accepted. Source integration, dependency reconciliation, image build and bounded 70B validation are authorized; rollout PP implementation is the next phase.

| Task/interface | Review | Decision |
|---|---|---|
| Task1 internally | Exact upstream baseline and import fix compatible in scope; optimizer fix must remain | Merge in standalone MCore repository |
| Task2 internally | New upstream Bridge includes approved correctness fixes and TE2.18 dependency update | Preserve Apertus fork while merging |
| Task3 internally | Existing successful runtime fixes are ahead of bumped image branch | Start from b7a6a2a61 |
| Task4 internally | Validation must not imply rollout PP is implemented | Use existing PP1 rollout path first |
| Task1 -> Task2 | MCore SHA consumed by Bridge .main.commit and gitlink | Both must agree |
| Task2 -> Task3 | Bridge SHA and dependency metadata consumed by NeMo-RL lock | Root integration waits for final Bridge commit |
| Task3 -> Task4 | Clean source and recursive pins determine image fingerprint | Rebuild dependencies; no fingerprint bypass |

Ruling: use repository-local .git/info/exclude for the worktree directory rather than modifying the user's dirty root .gitignore. This preserves the existing root source while avoiding accidental worktree staging.

Ruling: execute independent source integration locally while an isolated submodule implementer works. Only one implementation subagent at a time; all edits remain in disjoint repositories.

Status: isolated NeMo-RL worktree created; standalone Bridge/MCore clones created. No upstream changes merged yet. No new jobs submitted.
