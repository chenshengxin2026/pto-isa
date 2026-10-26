# GitCode synchronization

The publishing workflow runs in `chenshengxin2026/pto-isa`. Install the workflow
and `.github/scripts/sync_gitcode.py` on that fork's default branch. Scheduled
runs check GitCode `master` at 01:00 UTC on Monday-Friday; manual runs can select
branches, a full cut-point SHA, and dry-run mode. Publishing requires the fork's
`UPSTREAM_PR_TOKEN` secret. The workflow does not merge PRs.

## Historical dependencies

A merged PR's `Source SHA` identifies the processed source revision. It does not
prove that the GitHub tree contains every earlier GitCode dependency. Before
replaying changes, the script compares the GitHub base with the GitCode cut point
for paths touched by any replayed first-parent commit, including intermediate
edits reverted later in the range. When shared architecture headers change, it also
checks their architecture forwarding entries, even if those entries are absent
from the incoming delta. Missing historical literal `pto/` includes are restored
recursively, but unrelated existing dependencies are not replaced.

When these paths differ, the script starts a sync branch from the GitHub target
and restores their GitCode **cut-point** versions, including additions, deletions
and file modes. It records those changes in a dedicated historical baseline
commit, then replays incoming GitCode commits on top. Both parts are pushed
together and submitted as **one sync PR**. The PR body records the reconciliation
commit, affected paths, source SHA and generated head SHA. Existing GitHub edits
in the selected paths are intentionally replaced with the GitCode baseline;
unrelated target paths remain unchanged. The combined PR is the review boundary.

PR #368 exposed this distinction: GitCode's A5 `TCvt.hpp` already forwarded to
`common/arch/register/tcvt_common.hpp`, while GitHub still used the standalone
implementation. The fix adopts that forwarding entry so the common FP4 LUT
implementation and its tests use the same path.

Use `dry_run=true` to build both the baseline commit and incoming commits without
pushing or editing a PR. The include check covers literal `pto/` dependencies in incoming
code; it is not a compiler or a proof of semantic equivalence. A5 and CPU CI
remain required.

Before publication, selected paths must exactly match the source tip and
unrelated paths must still match the GitHub target. Whitespace checks accept
GitCode's existing CRLF line endings without rewriting files. Other whitespace
already present in the verified source is reported in the log and PR body, so
formatting alone does not prevent creating the combined PR. Added conflict
markers still stop publication. PR CI remains responsible for formatting checks.

## Branch and PR protection

- First-parent commits are replayed in order. Merge commits use their first
  parent; side-branch commits are not replayed a second time. A side-branch cut
  point is rejected.
- Only confirmed empty cherry-picks are skipped. Conflicts stop the run rather
  than automatically discarding target changes.
- Both `Source SHA` and `Sync head SHA` are refreshed in the PR body on every
  successful publication. The latest matching merged PR is selected using a
  paginated query filtered by target and source branch.
- An open PR whose head differs from its recorded generated head is preserved.
  Legacy PRs without a `Sync head SHA` are also preserved. Merge or reconcile
  those PRs manually before allowing automatic replacement; do not mark a manual
  fix as generated unless it is reproducible by the sync process.
- An explicit Git force-with-lease rejects concurrent branch changes. PR state,
  head and body are checked again before publication. If metadata publication
  fails after a push, the resulting marker mismatch stops the next run for review.
- No new commits or no net difference means no publication. A no-op does not
  automatically close an existing PR.

Historical alignment always uses the cut point, not the final source tip, so
incoming commits retain their order and messages. Reconciliation does not relax
the manual-PR or push-lease guards. Unexpected replay conflicts still stop before
publication. Diagnostics are retained as an Actions artifact.

Run the offline regression tests with:

```bash
python3 -m unittest discover -s .github/scripts -p 'test_sync_gitcode.py' -v
```
