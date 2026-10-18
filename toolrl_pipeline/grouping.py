"""Conservative leakage grouping; edges preserve evidence, never rewrite content."""
from collections import Counter, defaultdict
import re
import unicodedata

from .io import digest


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def user_turns(text):
    return [normalized(s) for s in re.findall(r"<user>(.*?)</user>", text, re.S)]


def last_query(text):
    turns = user_turns(text)
    return turns[-1] if turns else normalized(text)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def shingles(text):
    words = re.findall(r"\w+", normalized(text))
    return {tuple(words[i:i+3]) for i in range(len(words)-2)} if len(words) >= 8 else set()


def group_records(records, near_threshold=0.90):
    uf = UnionFind(len(records))
    buckets, edges = {}, []
    for i, row in enumerate(records):
        raw = row["raw"]
        turns = user_turns(raw["input"])
        keys = [("exact_prompt", digest([raw["instruction"], raw["input"]])),
                ("normalized_input", normalized(raw["input"]))]
        # Shared substantive utterances connect different turns of a conversation.
        keys += [("shared_history_utterance", t) for t in turns if len(t) >= 24]
        prefix = raw["input"].rsplit("<user>", 1)[0]
        if len(prefix) >= 100 and "<obs>" in prefix:
            keys.append(("history_prefix", normalized(prefix)))
        for kind, value in keys:
            key = (kind, value)
            if key in buckets:
                j = buckets[key]
                if uf.find(i) != uf.find(j):
                    edges.append({"a": records[j]["sample_id"], "b": row["sample_id"], "reason": kind})
                uf.union(i, j)
            else:
                buckets[key] = i
    # A longer history may contain an earlier row's complete input verbatim.
    # Sorting brings each prefix and its descendants together; no quadratic scan.
    stack = []
    for value, i in sorted((normalized(r['raw']['input']), i) for i, r in enumerate(records)):
        while stack and not value.startswith(stack[-1][0]):
            stack.pop()
        if stack and len(stack[-1][0]) >= 64:
            j = stack[-1][1]
            if uf.find(i) != uf.find(j):
                edges.append({'a': records[j]['sample_id'], 'b': records[i]['sample_id'], 'reason': 'complete_input_history_prefix'})
            uf.union(i, j)
        stack.append((value, i))
    query_sets = [shingles(last_query(r["raw"]["input"])) for r in records]
    inverted = defaultdict(list)
    for i, values in enumerate(query_sets):
        for value in values:
            inverted[value].append(i)
    candidates = set()
    for indices in inverted.values():
        # Very common template phrases do not generate quadratic candidates alone.
        if len(indices) > 200:
            continue
        for ii, i in enumerate(indices):
            for j in indices[ii+1:]:
                candidates.add((i, j))
    near_count = 0
    for i, j in sorted(candidates):
        a, b = query_sets[i], query_sets[j]
        if not a or not b:
            continue
        similarity = len(a & b) / len(a | b)
        if similarity >= near_threshold:
            near_count += 1
            if uf.find(i) != uf.find(j):
                edges.append({"a": records[i]["sample_id"], "b": records[j]["sample_id"],
                              "reason": "high_risk_query_near_duplicate", "jaccard": round(similarity, 6)})
            uf.union(i, j)
    members = defaultdict(list)
    for i in range(len(records)):
        members[uf.find(i)].append(records[i]["sample_id"])
    group_ids = {k: "g_"+digest(sorted(set(v)))[:24] for k, v in members.items()}
    for i, row in enumerate(records):
        row["group_id"] = group_ids[uf.find(i)]
    return {"edges": edges, "groups": len(members), "largest_group_occurrences": max(map(len, members.values()), default=0),
            "edge_reasons": dict(Counter(e["reason"] for e in edges)), "near_pairs_above_threshold": near_count,
            "near_threshold": near_threshold, "near_candidates": len(candidates),
            "limitations": ["Lexical grouping cannot prove absence of paraphrase or pretraining contamination.",
                            "Shingles occurring in >200 rows are excluded from candidate generation; substantive exact utterances are still grouped."]}


def assign_splits(records, seed):
    groups = defaultdict(list)
    for row in records:
        if row["action"] in ("keep", "repair"):
            groups[row["group_id"]].append(row)
    anchors = {gid for gid, rows in groups.items() if len({r["source"] for r in rows}) > 1}
    strata = defaultdict(list)
    for gid, rows in groups.items():
        source = "sft" if any(r["source"] == "sft" for r in rows) else "rl"
        task = Counter(r["metadata"]["task_type"] for r in rows).most_common(1)[0][0]
        strata[(source, task)].append((gid, rows))
    assignments = {}
    for _, groups_in_stratum in sorted(strata.items()):
        total = sum(len(rows) for _, rows in groups_in_stratum)
        targets = {"train": total*.8, "validation": total*.1, "test": total*.1}
        counts = Counter()
        for gid, rows in groups_in_stratum:
            if gid in anchors:
                assignments[gid] = "train"
                counts["train"] += len(rows)
        free = [(gid, rows) for gid, rows in groups_in_stratum if gid not in anchors]
        free.sort(key=lambda item: (-len(item[1]), digest([seed, item[0]])))
        for gid, rows in free:
            split = min(targets, key=lambda s: ((counts[s]+len(rows)/2)/max(targets[s], 1), s))
            assignments[gid] = split
            counts[split] += len(rows)
    for row in records:
        row["split"] = assignments.get(row["group_id"]) if row["action"] in ("keep", "repair") else None
    return {"seed": seed, "target_ratios": {"train": .8, "validation": .1, "test": .1},
            "anchored_cross_source_train_groups": len(anchors),
            "policy": "Keep cross-source retained groups in train; group-size-aware stratification of remaining groups; no row-wise split."}
