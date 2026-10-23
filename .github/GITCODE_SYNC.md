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
for touched `include/` paths. When shared architecture headers change, it also
checks their architecture forwarding entries, even if those entries are absent
from the incoming delta. Existing target corrections and missing historical
headers both cause a failure requiring review; files are not silently replaced.

PR #368 exposed this distinction: GitCode's A5 `TCvt.hpp` already forwarded to
`common/arch/register/tcvt_common.hpp`, while GitHub still used the standalone
implementation. The fix adopts that forwarding entry so the common FP4 LUT
implementation and its tests use the same path.

These checks are deliberately conservative. Reconcile the reported differences
in an upstream PR before retrying. A dry run applies the same guards and does not
bypass them. The include check covers literal `pto/` dependencies in incoming
code; it is not a compiler or a proof of semantic equivalence. A5 and CPU CI
remain required.

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

Do not remove historical baseline checks merely to get a scheduled run to pass.
When GitHub intentionally diverges from GitCode, either reconcile the dependency
or perform a reviewed manual port. Diagnostics are retained as an Actions artifact.

Run the offline regression tests with:

```bash
python3 -m unittest discover -s .github/scripts -p 'test_sync_gitcode.py' -v
```
