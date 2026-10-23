#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Replay GitCode changes only when their header baseline is compatible."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode


class SyncError(RuntimeError):
    pass


def command(*args, check=True):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise SyncError(f"{' '.join(args)}\n{result.stderr}")
    return result


def git(*args):
    return command("git", *args).stdout.strip()


def gh(*args):
    return json.loads(command("gh", *args).stdout)


def tree(ref):
    entries = {}
    for entry in git("ls-tree", "-r", "-z", ref).split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            entries[path] = metadata
    return entries


def changed_paths(cut, source):
    return set(filter(None, git("diff", "--name-only", "--no-renames", "-z", cut, source).split("\0")))


def check_baseline(base, cut, source):
    """Reject historical drift in touched headers and their forwarding entries.

    Compare against the source cut, not the new source tip. Incoming changes are
    expected to differ. Do not repair history by replacing arbitrary target files.
    """
    target_tree, cut_tree, source_tree = tree(base), tree(cut), tree(source)
    touched = changed_paths(cut, source)
    headers = {p for p in touched if p.startswith("include/")}
    # A shared implementation can apply while an old entry still bypasses it.
    for path in sorted(headers):
        if not path.startswith("include/pto/common/arch/") or path not in source_tree:
            continue
        matches = command(
            "git",
            "grep",
            "-l",
            "-F",
            "-e",
            path.removeprefix("include/"),
            source,
            "--",
            "include/pto/npu/",
            check=False,
        )
        if matches.returncode not in (0, 1):
            raise SyncError(matches.stderr)
        headers.update(line.split(":", 1)[1] for line in matches.stdout.splitlines())
    mismatches = [p for p in sorted(headers) if target_tree.get(p) != cut_tree.get(p)]
    if mismatches:
        details = "\n".join(
            f"- {p}: GitHub={target_tree.get(p, 'missing')}; GitCode cut={cut_tree.get(p, 'missing')}"
            for p in mismatches
        )
        raise SyncError(
            "Historical header baseline differs. Reconcile dependencies in a reviewed PR before syncing.\n" + details
        )
    return touched


def check_includes(source, paths):
    """Check literal PTO includes in incoming code against the result."""
    source_tree, result_tree = tree(source), tree("HEAD")
    missing = []
    for path in sorted(paths):
        if path not in source_tree or not path.endswith((".hpp", ".h", ".cpp", ".cc")):
            continue
        content = git("show", f"{source}:{path}")
        for dependency in re.findall(r'^\s*#\s*include\s*[<"](pto/[^>"\n]+)[>"]', content, re.MULTILINE):
            dep_path = "include/" + dependency
            if dep_path in source_tree and dep_path not in result_tree:
                missing.append(f"{path} -> {dep_path}")
    if missing:
        raise SyncError("Missing historical include dependencies:\n" + "\n".join(missing))


def recorded_sha(body, label):
    match = re.search(rf"^{re.escape(label)}:\s*`?([0-9a-fA-F]{{40}})`?", body or "", re.MULTILINE)
    return match.group(1).lower() if match else None


def guard_open_pr(pr):
    if pr and recorded_sha(pr["body"], "Sync head SHA") != pr["headRefOid"]:
        raise SyncError(
            f"PR #{pr['number']} has manual changes or no recorded generated head. "
            "Preserving its branch; reconcile or merge it before automatic replacement."
        )


def resolve_cut(upstream, base, head, source_ref, bootstrap):
    query = urlencode({"state": "closed", "base": base, "head": head, "per_page": 100})
    pages = gh("api", "--paginate", "--slurp", f"repos/{upstream}/pulls?{query}")
    merged = [pr for page in pages for pr in page if pr.get("merged_at")]
    owner, branch = head.split(":", 1)
    for pr in sorted(merged, key=lambda p: p["merged_at"], reverse=True):
        if pr["head"]["ref"] != branch or pr["head"]["user"]["login"].lower() != owner.lower():
            continue
        body = pr.get("body") or ""
        match = re.search(r"^Source ref:\s*`?([^`\s]+)", body, re.MULTILINE)
        if (match.group(1) if match else "master") != source_ref:
            continue
        sha = recorded_sha(body, "Source SHA")
        if sha:
            return sha, f"merged-pr-{pr['number']}"
    return bootstrap, "bootstrap-no-merged-pr"


def replay(cut, source, base):
    # Replay a merge relative to its first parent, without replaying its side
    # commits as well. Reject a cut on a side branch rather than guessing.
    chain = git("rev-list", "--first-parent", source).splitlines()
    if cut not in chain:
        raise SyncError("Cut point is not on the source first-parent chain; select a mainline cut point.")
    commits = list(reversed(chain[: chain.index(cut)]))
    git("checkout", "-B", "sync_work", base)
    skipped = 0
    for sha in commits:
        parents = git("rev-list", "--parents", "-n", "1", sha).split()[1:]
        args = ["git", "-c", "core.editor=true", "cherry-pick"]
        if len(parents) > 1:
            args += ["-m", "1"]
        result = command(*args, sha, check=False)
        if result.returncode == 0:
            continue
        active = command("git", "rev-parse", "--verify", "CHERRY_PICK_HEAD", check=False).returncode == 0
        unmerged = git("diff", "--name-only", "--diff-filter=U")
        clean = (
            command("git", "diff", "--quiet", check=False).returncode == 0
            and command("git", "diff", "--cached", "--quiet", check=False).returncode == 0
        )
        if active and not unmerged and clean:
            git("cherry-pick", "--skip")
            skipped += 1
            continue
        command("git", "cherry-pick", "--abort", check=False)
        raise SyncError(f"Cannot replay {sha}; no conflict was automatically overwritten.\n{result.stderr}")
    return len(commits), skipped


def validate_config():
    config = {
        "source": os.environ.get("INPUT_SOURCE_REF") or "master",
        "target": os.environ.get("INPUT_TARGET_BRANCH") or "main",
        "branch": os.environ.get("INPUT_SYNC_BRANCH") or "gitcode-sync",
        "cut": os.environ.get("INPUT_CUT_POINT") or "auto",
        "dry": os.environ.get("INPUT_DRY_RUN") or "false",
    }
    for name in ("source", "target", "branch"):
        command("git", "check-ref-format", "--branch", config[name])
        if config[name].startswith("-") or any(c.isspace() for c in config[name]):
            raise SyncError(f"Invalid branch: {name}")
    if config["branch"] in (config["target"], os.environ.get("DEFAULT_BRANCH", "main")):
        raise SyncError("The sync branch must not be the target or fork default branch.")
    if config["cut"] != "auto" and not re.fullmatch(r"[0-9a-fA-F]{40}", config["cut"]):
        raise SyncError("cut_point must be auto or a full 40-character commit SHA.")
    if config["dry"] not in ("true", "false"):
        raise SyncError("dry_run must be true or false.")
    return config


def sync():
    config = validate_config()
    upstream = os.environ["GITHUB_UPSTREAM"]
    head = f"{os.environ['GITHUB_REPOSITORY_OWNER']}:{config['branch']}"
    publishing = config["dry"] == "false"
    if publishing and not os.environ.get("UPSTREAM_PR_TOKEN"):
        raise SyncError("Missing UPSTREAM_PR_TOKEN for branch push and cross-repository PR publication.")
    open_prs = gh(
        "pr",
        "list",
        "--repo",
        upstream,
        "--base",
        config["target"],
        "--head",
        head,
        "--state",
        "open",
        "--json",
        "number,url,body,headRefOid",
    )
    if len(open_prs) > 1:
        raise SyncError("Multiple matching open PRs; review them before syncing.")
    existing = open_prs[0] if open_prs else None
    guard_open_pr(existing)
    cut, cut_source = config["cut"], "manual"
    if cut == "auto":
        cut, cut_source = resolve_cut(
            upstream, config["target"], head, config["source"], os.environ["BOOTSTRAP_CUT_POINT"]
        )
    git("fetch", "--no-tags", f"https://github.com/{upstream}.git", f"refs/heads/{config['target']}")
    base = git("rev-parse", "FETCH_HEAD")
    git("fetch", "--no-tags", os.environ["GITCODE_UPSTREAM_URL"], f"refs/heads/{config['source']}")
    source = git("rev-parse", "FETCH_HEAD")
    cut = git("rev-parse", "--verify", f"{cut}^{{commit}}")
    git("merge-base", "--is-ancestor", cut, source)
    print(f"GitHub base: {base}\nSource SHA: {source}\nCut point: {cut} ({cut_source})", flush=True)
    if cut == source:
        print("No new source commits.")
        return
    # Capture before building; an explicit empty lease also protects branch creation.
    remote = git("ls-remote", "--heads", "origin", f"refs/heads/{config['branch']}")
    expected_head = remote.split()[0] if remote else ""
    if existing and expected_head != existing["headRefOid"]:
        raise SyncError("PR head changed during planning; retry without overwriting it.")
    paths = check_baseline(base, cut, source)
    count, skipped = replay(cut, source, base)
    check_includes(source, paths)
    git("diff", "--check", base, "HEAD")
    generated = git("rev-parse", "HEAD")
    print(f"First-parent commits: {count}; skipped empty: {skipped}; sync head: {generated}", flush=True)
    if command("git", "diff", "--quiet", base, "HEAD", check=False).returncode == 0:
        print("No net change; existing PR, if any, is left for review.")
        return
    if not publishing:
        print("Dry run completed; no branch or PR was updated.")
        return
    if existing:
        current = gh("pr", "view", str(existing["number"]), "--repo", upstream, "--json", "state,headRefOid,body")
        if current["state"] != "OPEN" or current["headRefOid"] != expected_head or current["body"] != existing["body"]:
            raise SyncError("PR changed during the build; preserving it for review.")
    git(
        "push",
        f"--force-with-lease=refs/heads/{config['branch']}:{expected_head}",
        "origin",
        f"HEAD:refs/heads/{config['branch']}",
    )
    title = f"sync: GitCode {config['source']} @ {source[:12]}"
    body = (
        f"Automated sync from GitCode `cann/pto-isa:{config['source']}` into GitHub `{upstream}:{config['target']}`.\n\n"
        f"Carries only commits after cut point:\n`{cut}`\n\nCut point source:\n`{cut_source}`\n\n"
        f"Source ref:\n`{config['source']}`\n\nSource SHA:\n`{source}`\n\n"
        f"Sync head SHA:\n`{generated}`\n\nSync mode:\n`first-parent-cherry-pick`\n\n"
        "Historical baseline checks passed for touched headers and their architecture forwarding entries. "
        "GitHub CI still needs to pass before this PR is merged.\n"
    )
    body_file = Path(os.environ["RUNNER_TEMP"]) / "gitcode-sync-pr.md"
    body_file.write_text(body)
    if existing:
        result = command(
            "gh",
            "pr",
            "edit",
            str(existing["number"]),
            "--repo",
            upstream,
            "--title",
            title,
            "--body-file",
            str(body_file),
        )
    else:
        result = command(
            "gh",
            "pr",
            "create",
            "--repo",
            upstream,
            "--base",
            config["target"],
            "--head",
            head,
            "--title",
            title,
            "--body-file",
            str(body_file),
        )
    print(result.stdout)


if __name__ == "__main__":
    try:
        sync()
    except (SyncError, KeyError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        sys.exit(1)
