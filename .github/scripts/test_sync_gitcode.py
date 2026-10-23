# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the sync guards and replay against real, isolated Git histories."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("sync_gitcode", Path(__file__).with_name("sync_gitcode.py"))
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.previous = Path.cwd()
        self.repo = root / "repo"
        self.repo.mkdir()
        os.chdir(self.repo)
        self.addCleanup(os.chdir, self.previous)
        sync.git("init", "-b", "main")
        sync.git("config", "user.name", "Sync Test")
        sync.git("config", "user.email", "sync@example.invalid")
        self.write("include/pto/example.hpp", "// baseline\n")
        self.cut = self.commit("baseline")
        self.remote = str(root / "remote.git")
        sync.git("clone", "--bare", str(self.repo), self.remote)
        sync.git("remote", "add", "origin", self.remote)

    def write(self, path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def commit(self, message):
        sync.git("add", "-A")
        sync.git("commit", "-m", message)
        return sync.git("rev-parse", "HEAD")

    def source_change(self):
        sync.git("checkout", "-b", "source", self.cut)
        self.write("include/pto/example.hpp", "// incoming\n")
        return self.commit("incoming")

    def test_matching_baseline_and_replay(self):
        source = self.source_change()
        paths = sync.check_baseline(self.cut, self.cut, source)
        self.assertEqual(paths, {"include/pto/example.hpp"})
        self.assertEqual(sync.replay(self.cut, source, self.cut), (1, 0))
        self.assertEqual(sync.git("rev-parse", "HEAD^{tree}"), sync.git("rev-parse", f"{source}^{{tree}}"))

    def test_historical_drift_stops_before_checkout(self):
        source = self.source_change()
        sync.git("checkout", "-b", "target", self.cut)
        self.write("include/pto/example.hpp", "// divergent target\n")
        base = self.commit("target-only")
        with self.assertRaisesRegex(sync.SyncError, "Historical header baseline differs"):
            sync.check_baseline(base, self.cut, source)
        self.assertEqual(sync.git("rev-parse", "HEAD"), base)

    def test_unchanged_architecture_entry_is_checked(self):
        wrapper = "include/pto/npu/a5/TCvt.hpp"
        shared = "include/pto/common/arch/register/tcvt_common.hpp"
        self.write(shared, "// shared baseline\n")
        self.write(wrapper, '#include "pto/common/arch/register/tcvt_common.hpp"\n')
        cut = self.commit("historical move to shared implementation")
        self.write(shared, "// new shared feature\n")
        source = self.commit("feature")
        sync.git("checkout", "-b", "legacy-target", cut)
        self.write(wrapper, "// old standalone implementation\n")
        base = self.commit("old entry still exists")
        with self.assertRaisesRegex(sync.SyncError, wrapper):
            sync.check_baseline(base, cut, source)

    def test_missing_historical_shared_header_is_checked(self):
        self.write("include/pto/common/arch/register/operation.hpp", "// baseline\n")
        cut = self.commit("historical shared header")
        self.write("include/pto/common/arch/register/operation.hpp", "// feature\n")
        source = self.commit("feature")
        with self.assertRaisesRegex(sync.SyncError, "GitHub=missing"):
            sync.check_baseline(self.cut, cut, source)

    def test_merge_replays_side_change_exactly_once(self):
        sync.git("checkout", "-b", "side", self.cut)
        self.write("side.txt", "side\n")
        side = self.commit("side change")
        sync.git("checkout", "-b", "source", self.cut)
        self.write("mainline.txt", "mainline\n")
        self.commit("mainline change")
        sync.git("merge", "--no-ff", "side", "-m", "merge side")
        source = sync.git("rev-parse", "HEAD")
        self.assertEqual(sync.replay(self.cut, source, self.cut), (2, 0))
        self.assertEqual(Path("side.txt").read_text(), "side\n")
        self.assertEqual(sync.git("rev-parse", "HEAD^{tree}"), sync.git("rev-parse", f"{source}^{{tree}}"))
        with self.assertRaisesRegex(sync.SyncError, "first-parent"):
            sync.replay(side, source, self.cut)

    def test_conflicts_are_not_resolved_by_overwriting_target(self):
        source = self.source_change()
        sync.git("checkout", "-b", "target", self.cut)
        self.write("include/pto/example.hpp", "// target correction\n")
        base = self.commit("target correction")
        with self.assertRaisesRegex(sync.SyncError, "Cannot replay"):
            sync.replay(self.cut, source, base)
        self.assertEqual(sync.git("rev-parse", "HEAD"), base)
        self.assertEqual(Path("include/pto/example.hpp").read_text(), "// target correction\n")
        self.assertEqual(sync.git("status", "--porcelain"), "")

    def test_already_present_patch_is_skipped(self):
        source = self.source_change()
        self.assertEqual(sync.replay(self.cut, source, source), (1, 1))

    def test_missing_literal_include_is_rejected(self):
        self.write("include/pto/dependency.hpp", "// dependency\n")
        dependency = self.commit("historical dependency")
        self.write("example.cpp", '#include "pto/dependency.hpp"\n')
        source = self.commit("consumer")
        sync.replay(dependency, source, self.cut)
        with self.assertRaisesRegex(sync.SyncError, "Missing historical include"):
            sync.check_includes(source, {"example.cpp"})

    def test_manual_or_legacy_pr_is_preserved(self):
        pr = {"number": 368, "body": "Source SHA:\n`" + self.cut + "`", "headRefOid": self.cut}
        with self.assertRaisesRegex(sync.SyncError, "Preserving its branch"):
            sync.guard_open_pr(pr)
        pr["body"] += "\nSync head SHA:\n`" + self.cut + "`"
        sync.guard_open_pr(pr)
        pr["headRefOid"] = "a" * 40
        with self.assertRaises(sync.SyncError):
            sync.guard_open_pr(pr)

    def test_invalid_inputs_cannot_target_the_default_branch(self):
        for values in (
            {"INPUT_SYNC_BRANCH": "main"},
            {"INPUT_SOURCE_REF": "bad branch"},
            {"INPUT_CUT_POINT": "HEAD~3"},
            {"INPUT_DRY_RUN": "yes"},
        ):
            with self.subTest(values=values), patch.dict(os.environ, values, clear=True):
                with self.assertRaises(sync.SyncError):
                    sync.validate_config()

    def test_cherry_pick_failure_without_active_pick_is_not_skipped(self):
        source = self.source_change()
        original = sync.command

        def fail_pick(*args, **kwargs):
            if "cherry-pick" in args and "--abort" not in args:
                return sync.subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected failure")
            return original(*args, **kwargs)

        with patch.object(sync, "command", side_effect=fail_pick):
            with self.assertRaisesRegex(sync.SyncError, "unexpected failure"):
                sync.replay(self.cut, source, self.cut)

    def test_paginated_cut_uses_latest_matching_merged_pr(self):
        def pr(number, date, branch="gitcode-sync", source_ref="master"):
            return {
                "number": number,
                "merged_at": date,
                "head": {"ref": branch, "user": {"login": "fork"}},
                "body": f"Source SHA:\n`{number:040x}`\nSource ref:\n`{source_ref}`",
            }

        pages = [[pr(1, "2026-01-01"), pr(3, "2026-01-03", source_ref="other")], [pr(2, "2026-01-02")]]
        with patch.object(sync, "gh", return_value=pages) as api:
            self.assertEqual(
                sync.resolve_cut("up/repo", "main", "fork:gitcode-sync", "master", self.cut),
                (f"{2:040x}", "merged-pr-2"),
            )
        self.assertIn("--paginate", api.call_args.args)

    def exercise_publication(self, dry, existing=None):
        self.source_change()
        original_git, original_command = sync.git, sync.command
        edits = []

        def git(*args):
            if args[0] == "fetch":
                ref = "main" if args[2].startswith("https://github.com/") else "source"
                return original_git("fetch", "--no-tags", str(self.repo), ref)
            return original_git(*args)

        def gh(*args):
            if args[:2] == ("pr", "list"):
                return [existing] if existing else []
            if args[:2] == ("pr", "view"):
                return {"state": "OPEN", "headRefOid": existing["headRefOid"], "body": existing["body"]}
            raise AssertionError(args)

        def command(*args, **kwargs):
            if args[0] == "gh":
                edits.append((args, Path(args[-1]).read_text()))
                return subprocess_result()
            return original_command(*args, **kwargs)

        env = {
            "INPUT_CUT_POINT": self.cut,
            "INPUT_DRY_RUN": dry,
            "UPSTREAM_PR_TOKEN": "test-token",
            "GITHUB_UPSTREAM": "up/repo",
            "GITHUB_REPOSITORY_OWNER": "fork",
            "GITCODE_UPSTREAM_URL": "local-source",
            "RUNNER_TEMP": self.temp.name,
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(sync, "git", side_effect=git),
            patch.object(sync, "gh", side_effect=gh),
            patch.object(sync, "command", side_effect=command),
        ):
            sync.sync()
        return edits

    def test_dry_run_does_not_publish(self):
        self.assertEqual(self.exercise_publication("true"), [])
        self.assertEqual(sync.git("ls-remote", "--heads", "origin", "refs/heads/gitcode-sync"), "")

    def test_publish_records_source_and_generated_head(self):
        edits = self.exercise_publication("false")
        head = sync.git("rev-parse", "HEAD")
        self.assertIn(head, sync.git("ls-remote", "--heads", "origin", "refs/heads/gitcode-sync"))
        self.assertEqual(edits[0][0][1:3], ("pr", "create"))
        self.assertEqual(sync.recorded_sha(edits[0][1], "Source SHA"), sync.git("rev-parse", "source"))
        self.assertEqual(sync.recorded_sha(edits[0][1], "Sync head SHA"), head)

    def test_existing_pr_metadata_is_refreshed(self):
        sync.git("push", "origin", f"{self.cut}:refs/heads/gitcode-sync")
        existing = {"number": 368, "body": f"Sync head SHA:\n`{self.cut}`", "headRefOid": self.cut}
        edits = self.exercise_publication("false", existing)
        self.assertEqual(edits[0][0][1:4], ("pr", "edit", "368"))
        self.assertEqual(sync.recorded_sha(edits[0][1], "Source SHA"), sync.git("rev-parse", "source"))

    def test_lease_rejects_concurrent_branch_update(self):
        sync.git("push", "origin", f"{self.cut}:refs/heads/gitcode-sync")
        source = self.source_change()
        sync.git("push", "origin", f"{source}:refs/heads/gitcode-sync")
        with self.assertRaises(sync.SyncError):
            sync.git(
                "push",
                f"--force-with-lease=refs/heads/gitcode-sync:{self.cut}",
                "origin",
                f"{self.cut}:refs/heads/gitcode-sync",
            )


def subprocess_result():
    return sync.subprocess.CompletedProcess(["gh"], 0, stdout="https://example.invalid/pr/368", stderr="")


if __name__ == "__main__":
    unittest.main()
