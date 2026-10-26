#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reconcile the GitCode baseline, then publish it and incoming commits in one PR."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode


class SyncError(RuntimeError):
    pass


def command(*args, check=True):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise SyncError(f"{' '.join(args)}\n{result.stdout}{result.stderr}")
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


def first_parent_commits(cut, source):
    chain = git("rev-list", "--first-parent", source).splitlines()
    if cut not in chain:
        raise SyncError("Cut point is not on the source first-parent chain; select a mainline cut point.")
    return list(reversed(chain[: chain.index(cut)]))


def baseline_paths(base, cut, source):
    """Find replayed paths, architecture entries, and missing historical includes.

    Include intermediate edits that cancel out at the source tip: they still
    need the correct baseline when their commits are replayed in sequence.
    """
    target_tree, cut_tree, source_tree = tree(base), tree(cut), tree(source)
    paths = set()
    for sha in first_parent_commits(cut, source):
        paths.update(changed_paths(f"{sha}^1", sha))
    # A shared implementation can apply while an old entry still bypasses it.
    for path in sorted(paths):
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
        paths.update(line.split(":", 1)[1] for line in matches.stdout.splitlines())
    # Recover dependencies predating the cut that GitHub never received. Do not
    # replace unrelated existing headers just because a consumer includes them.
    pending = list(paths)
    while pending:
        path = pending.pop()
        if not path.endswith((".hpp", ".h", ".cpp", ".cc")):
            continue
        for ref, entries in ((cut, cut_tree), (source, source_tree)):
            if path not in entries or " blob " not in entries[path]:
                continue
            for dependency in literal_includes(git("show", f"{ref}:{path}")):
                if dependency in cut_tree and dependency not in target_tree and dependency not in paths:
                    paths.add(dependency)
                    pending.append(dependency)
    return paths


def reconcile_baseline(base, cut, source):
    """Commit the relevant GitCode cut-point versions before replaying new work."""
    paths = baseline_paths(base, cut, source)
    target_tree, cut_tree = tree(base), tree(cut)
    mismatches = [p for p in sorted(paths) if target_tree.get(p) != cut_tree.get(p)]
    git("checkout", "-B", "sync_work", base)
    reconciliation = None
    if mismatches:
        print(f"Reconciling {len(mismatches)} historical paths to GitCode cut {cut}:", flush=True)
        for path in mismatches:
            print(f"- {path}: GitHub={target_tree.get(path, 'missing')}; GitCode cut={cut_tree.get(path, 'missing')}")
        # Delete absent paths first, then restore existing cut paths. This also
        # supports file/directory transitions without command-line length limits.
        for present in (False, True):
            selected = [path for path in mismatches if (path in cut_tree) == present]
            if not selected:
                continue
            with tempfile.NamedTemporaryFile() as pathspec:
                pathspec.write(b"".join(f":(literal){path}\0".encode() for path in selected))
                pathspec.flush()
                operation = ["restore", f"--source={cut}", "--staged", "--worktree"] if present else ["rm"]
                git(*operation, f"--pathspec-from-file={pathspec.name}", "--pathspec-file-nul")
        git(
            "commit",
            "-m",
            f"sync: reconcile historical GitCode baseline at {cut[:12]}",
            "-m",
            f"Restore relevant paths to source cut {cut} before replaying incoming commits.\n\n"
            + "\n".join(f"- {path}" for path in mismatches),
        )
        reconciliation = git("rev-parse", "HEAD")
        print(f"Baseline reconciliation commit: {reconciliation}", flush=True)
    return paths, mismatches, reconciliation


def literal_includes(content):
    return {
        "include/" + dependency
        for dependency in re.findall(r'^\s*#\s*include\s*[<"](pto/[^>"\n]+)[>"]', content, re.MULTILINE)
    }


def check_includes(source, paths):
    """Check literal PTO includes in incoming code against the result."""
    source_tree, result_tree = tree(source), tree("HEAD")
    missing = []
    for path in sorted(paths):
        if path not in source_tree or not path.endswith((".hpp", ".h", ".cpp", ".cc")):
            continue
        content = git("show", f"{source}:{path}")
        for dep_path in sorted(literal_includes(content)):
            if dep_path in source_tree and dep_path not in result_tree:
                missing.append(f"{path} -> {dep_path}")
    if missing:
        raise SyncError("Missing historical include dependencies:\n" + "\n".join(missing))


def validate_result(base, source, paths):
    """Verify source alignment and preserve unrelated target paths before pushing."""
    result_tree, source_tree, base_tree = tree("HEAD"), tree(source), tree(base)
    mismatches = sorted(p for p in paths if result_tree.get(p) != source_tree.get(p))
    if mismatches:
        raise SyncError("Replayed paths differ from GitCode source:\n" + "\n".join(mismatches))
    unrelated = sorted(
        p for p in result_tree.keys() | base_tree.keys() if p not in paths and result_tree.get(p) != base_tree.get(p)
    )
    if unrelated:
        raise SyncError("Unrelated target paths changed:\n" + "\n".join(unrelated))
    check_includes(source, paths)
    # Preserve verified source bytes, including CRLF and historical whitespace.
    # Conflict markers still block publication; formatting is reported for PR CI.
    whitespace = command(
        "git",
        "-c",
        "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol",
        "diff",
        "--check",
        base,
        "HEAD",
        check=False,
    )
    if whitespace.returncode:
        if whitespace.returncode != 2:
            raise SyncError(f"Cannot check result whitespace:\n{whitespace.stdout}{whitespace.stderr}")
        git("-c", "core.whitespace=-blank-at-eol,-blank-at-eof,-space-before-tab", "diff", "--check", base, "HEAD")
        print(
            "::warning::Preserving whitespace already present in the verified GitCode source; review formatting in PR CI."
        )
        print(whitespace.stdout, flush=True)
    return bool(whitespace.returncode)


def recorded_sha(body, label):
    match = re.search(rf"^{re.escape(label)}:\s*`?([0-9a-fA-F]{{40}})`?", body or "", re.MULTILINE)
    return match.group(1).lower() if match else None


def guard_open_pr(pr):
    if pr and recorded_sha(pr["body"], "Sync head SHA") != pr["headRefOid"]:
        raise SyncError(
            f"PR #{pr['number']} has manual changes or no recorded generated head. "
            "Preserving its branch; reconcile or merge it before automatic replacement."
        )


def matching_prs(upstream, base, head, state):
    # gh pr list --head does not support owner:branch. The REST endpoint does.
    query = urlencode({"state": state, "base": base, "head": head, "per_page": 100})
    pages = gh("api", "--paginate", "--slurp", f"repos/{upstream}/pulls?{query}")
    owner, branch = head.split(":", 1)
    return [
        pr
        for page in pages
        for pr in page
        if pr["head"]["ref"] == branch and pr["head"]["user"]["login"].lower() == owner.lower()
    ]


def resolve_cut(upstream, base, head, source_ref, bootstrap):
    merged = [pr for pr in matching_prs(upstream, base, head, "closed") if pr.get("merged_at")]
    for pr in sorted(merged, key=lambda p: p["merged_at"], reverse=True):
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
    commits = first_parent_commits(cut, source)
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
    open_prs = [
        {"number": pr["number"], "body": pr.get("body") or "", "headRefOid": pr["head"]["sha"]}
        for pr in matching_prs(upstream, config["target"], head, "open")
    ]
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
    paths, reconciled_paths, reconciliation = reconcile_baseline(base, cut, source)
    count, skipped = replay(cut, source, reconciliation or base)
    source_whitespace = validate_result(base, source, paths)
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
    reconciliation_details = (
        f"Historical baseline commit:\n`{reconciliation}`\n\n"
        f"Aligned {len(reconciled_paths)} paths to the GitCode cut-point versions before replay:\n"
        + "\n".join(f"- `{path}`" for path in reconciled_paths)
        + "\n\n"
        if reconciliation
        else "Historical baseline reconciliation: no changes needed.\n\n"
    )
    formatting_note = (
        "Source formatting: GitCode whitespace was preserved unchanged; diagnostics are in the workflow log.\n\n"
        if source_whitespace
        else ""
    )
    body = (
        f"Automated sync from GitCode `cann/pto-isa:{config['source']}` into GitHub `{upstream}:{config['target']}`.\n\n"
        "Includes historical baseline reconciliation and subsequent source commits in this single PR.\n\n"
        f"Source replay starts after cut point:\n`{cut}`\n\nCut point source:\n`{cut_source}`\n\n"
        f"Source ref:\n`{config['source']}`\n\nSource SHA:\n`{source}`\n\n"
        f"Sync head SHA:\n`{generated}`\n\nSync mode:\n`baseline-reconciliation-and-first-parent-cherry-pick`\n\n"
        f"{reconciliation_details}"
        f"Replayed source commits: {count}; skipped empty: {skipped}.\n\n"
        f"{formatting_note}"
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
