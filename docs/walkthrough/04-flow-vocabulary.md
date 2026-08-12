# 04 — Flow: vocabulary build

**Entry:** [`src/scripts/build_vocabulary.py`](../../src/scripts/build_vocabulary.py)
**In:** corpus gene list + pre-downloaded pathway data
**Out:** `vocabulary.json`

This is where `D` and `K` are decided, and therefore where model size and the
per-group attention cost are decided.

---

## 0. Five modes — `main` [`:762`](../../src/scripts/build_vocabulary.py:762)

`main` is a dispatcher. It checks in this order and **returns early** from each:

| Mode | Trigger | Network? | Output |
|---|---|---|---|
| save-hierarchy | `--save-hierarchy DIR` | yes | Reactome + GO structure files |
| synthetic | `--use-synthetic` | no | fake vocabulary for tests |
| save-gene-sets | `--save-gene-sets F` | yes | `gene_sets.json` |
| hierarchical | `--grouping hierarchical` | no | `vocabulary.json` ← **live path** |
| greedy | `--grouping greedy` (default) | no | `vocabulary.json` |

> **Decision:** the two download modes exist as separate modes that exit
> immediately because BlueBEAR compute nodes are firewalled. You run them once on
> a login node; every later build reads from disk. This is the reason the build
> is split at all.

The guard at [`:774`](../../src/scripts/build_vocabulary.py:774) requires `--out`
unless you are in a mode that does not produce a vocabulary.

---

## 1. Fetching (login node only)

- `fetch_hallmarks` [`:255`](../../src/scripts/build_vocabulary.py:255) — MSigDB
  Hallmark sets, via `_fetch_hallmarks_from_enrichr` [`:207`](../../src/scripts/build_vocabulary.py:207)
  with `_fetch_one_hallmark_from_msigdb` [`:238`](../../src/scripts/build_vocabulary.py:238)
  as a per-set retrying fallback.
- `fetch_enrichr_library` [`:279`](../../src/scripts/build_vocabulary.py:279) —
  any named Enrichr library.
- `save_gene_sets` / `load_gene_sets` [`:582`](../../src/scripts/build_vocabulary.py:582),
  [`:594`](../../src/scripts/build_vocabulary.py:594) — the disk round-trip.
- `save_hierarchy_files` [`fetch.py:37`](../../src/endo_model/data/grouping/fetch.py:37)
  — downloads `ReactomePathwaysRelation.txt`, `ReactomePathways.txt`,
  `go-basic.obo` via `_download` [`:29`](../../src/endo_model/data/grouping/fetch.py:29).

**Why both** an Enrichr library *and* the structure files: Enrichr gives
gene→pathway membership but no parent/child edges; the Reactome/GO files give
the DAG but no gene lists. The join key is the accession embedded in the Enrichr
term name — `"Immune System R-HSA-168256"`, `"regulation of apoptosis (GO:0042981)"`.

---

## 2. The hierarchical path

### `build_hierarchical_vocabulary` — [`:517`](../../src/scripts/build_vocabulary.py:517)

```python
reactome = load_reactome(hdir/"ReactomePathwaysRelation.txt", hdir/"ReactomePathways.txt", reactome_enrichr)
go       = load_go(hdir/"go-basic.obo", go_enrichr)
groups   = build_hierarchical_groups(set(gene_list), reactome, go, strategy)
return emit_vocabulary(groups, gene_list)
```

### `load_reactome` — [`reactome.py:34`](../../src/endo_model/data/grouping/reactome.py:34)

1. Parse `ReactomePathwaysRelation.txt` (parent⇥child), keeping only rows where
   **both** ids start `R-HSA-` — the file is all species, and a mouse subtree
   would silently import mouse gene sets.
2. Parse `ReactomePathways.txt` for labels, filtered to `Homo sapiens`.
3. For each Enrichr set, `_extract_stable_id` [`:29`](../../src/endo_model/data/grouping/reactome.py:29)
   pulls `R-HSA-\d+` from the name; genes become that node's `own_genes`.
4. Register every referenced node in all four dicts so no lookup can `KeyError`.

Real numbers: **2,897 nodes, 43 roots, 1,818 gene sets matched, 0 unmatched.**

### `load_go` — [`go.py:84`](../../src/endo_model/data/grouping/go.py:84)

`_parse_obo` [`:31`](../../src/endo_model/data/grouping/go.py:31) is a
hand-rolled OBO stanza parser using a `flush()` closure
([`:48`](../../src/endo_model/data/grouping/go.py:48)) called on each `[Term]`
boundary and at EOF. It keeps a term only if `namespace == biological_process`
and `not is_obsolete`.

Edges come from **two** line types:

```
is_a: GO:0008150
relationship: part_of GO:0008150
```

Real numbers: **24,516 terms, 388 roots.**

> Those 388 roots are the single most consequential fact in this file. GO
> biological_process has exactly *one* true root (`GO:0008150`). The other 387
> are terms whose only parents come through relationship types this parser does
> not read (`regulates`, `negatively_regulates`, …), so they surface as
> parentless. Any logic that assumes "GO is single-rooted" is wrong — which is
> precisely what broke `--go-groups`. See §4.

### `Hierarchy` — [`hierarchy.py:14`](../../src/endo_model/data/grouping/hierarchy.py:14)

One source-agnostic DAG class, so the cut algorithms never branch on data
source. Five fields: `name`, `labels`, `children`, `parents`, `own_genes`.

- `roots()` [`:45`](../../src/endo_model/data/grouping/hierarchy.py:45) — nodes
  with no parents, **sorted** for determinism.
- `descendants(node)` [`:52`](../../src/endo_model/data/grouping/hierarchy.py:52)
  — iterative DFS with a `seen` set (a DAG reaches the same node by many paths),
  memoised.
- `node_genes(node)` [`:68`](../../src/endo_model/data/grouping/hierarchy.py:68)
  — own genes ∪ every descendant's own genes, memoised.

Both caches are why a cut that calls `node_size` thousands of times stays fast.

---

## 3. Frontier primitives — [`frontier.py`](../../src/endo_model/data/grouping/frontier.py)

A **frontier** is an antichain: a set of nodes such that every gene can be
attributed to exactly one. Both cuts differ only in *which* nodes end up on it;
descent and attribution are shared here so they cannot drift apart.

| Function | Line | Role |
|---|---|---|
| `node_size` | [`:25`](../../src/endo_model/data/grouping/frontier.py:25) | genes under a node ∩ universe |
| `root_frontier` | [`:30`](../../src/endo_model/data/grouping/frontier.py:30) | initial frontier = gene-bearing roots |
| `gene_bearing_children` | [`:35`](../../src/endo_model/data/grouping/frontier.py:35) | children with ≥1 universe gene |
| `expand_node` | [`:45`](../../src/endo_model/data/grouping/frontier.py:45) | replace a node with its children |
| `ancestors` | [`:71`](../../src/endo_model/data/grouping/frontier.py:71) | node + everything above it |
| `nearest_common_ancestor` | [`:92`](../../src/endo_model/data/grouping/frontier.py:92) | smallest shared ancestor, or `None` |
| `expand_largest` | [`:114`](../../src/endo_model/data/grouping/frontier.py:114) | expand biggest candidate; `False` if none |
| `assign_most_specific` | [`:138`](../../src/endo_model/data/grouping/frontier.py:138) | final gene → group attribution |

### `expand_node` and the `::other` groups

Descending a node promotes its children — but genes annotated **directly** to the
parent and to no child would vanish. `expand_node` computes that leftover and
records it as `<node>::other`. These are the "known broad function, unknown
specific pathway" genes.

### `assign_most_specific` — the attribution rule

A gene reachable from several frontier nodes (multi-parent DAG) goes to the
**smallest** one. Implementation: iterate frontier nodes sorted ascending by
`(node_size, id)` and use `setdefault`, so the first — smallest — writer wins.
Leftovers are applied afterwards, also with `setdefault`, so they only take
genes no frontier node claimed.

> **Decision:** smallest-wins means "most specific description available". Ties
> break on node id so the result is deterministic across runs.

### `ancestors` and `::other`

`ancestors` strips the `::other` suffix before walking
([`:79`](../../src/endo_model/data/grouping/frontier.py:79)). A residual group
sits *at* its node, so it is as closely related to that node's siblings as any
child is — which is what makes it merge sensibly in §4.

---

## 4. The two cuts

Both are `CutStrategy` implementations ([`strategies.py`](../../src/endo_model/data/grouping/strategies.py))
selected by `--cut`. They control **opposite ends** of the size distribution.

### `tree_cut` — a ceiling — [`tree_cut.py:46`](../../src/endo_model/data/grouping/tree_cut.py:46)

Loop `expand_largest` over frontier nodes exceeding `target_size` until none can
be expanded, then attribute.

**A ceiling cannot control a median.** Expanding one oversized node promotes
*all* its children at once, so a 600-gene node becomes fifteen 40-gene groups.
Measured on the real corpus at `target_size=500`: **202 groups, median 60** —
every group under the bound, almost none near it.

### `union_cut` — a floor, plus a ceiling — [`union_cut.py:48`](../../src/endo_model/data/grouping/union_cut.py:48)

Five phases:

```
1. frontier = gene-bearing roots
2. ceiling pass: expand any node with node_size > max_size
3. count pass:   expand while len(frontier) < expand_to
4. assign_most_specific
5. floor pass:   merge_undersized
```

**Phase 3 is nearly always inert**, and this is worth internalising. `expand_to`
is a *frontier count*: `while expand_to is not None and len(frontier) < expand_to`.
GO parses to 388 roots (§2), so with `--go-groups 4` the condition is false
immediately and the loop never runs. `GO:0008150` then sat on the frontier as a
peer of 387 pseudo-roots and, under `assign_most_specific`, collected every gene
no smaller node claimed — **3,869 of them**, which is 123 GB of attention at
batch 256. Phase 2 (`max_size`) is the knob that actually bites.

The warning at [`:76`](../../src/endo_model/data/grouping/union_cut.py:76) fires
if `max_size < 2 * min_size`, because phase 5 can produce groups near twice the
floor — a tighter ceiling cannot hold.

### `merge_undersized` — [`union_cut.py:127`](../../src/endo_model/data/grouping/union_cut.py:127)

While the smallest group is under `min_size`, merge it into the partner chosen
by `_pick_partner`.

Ancestry is tracked incrementally: `ancestry[g]` starts as that node's ancestor
set, and on merge becomes the **intersection** of both — which is exactly the
common-ancestor set of the merged group, maintained in O(1) per merge instead of
recomputed.

### `_pick_partner` — [`union_cut.py:170`](../../src/endo_model/data/grouping/union_cut.py:170)

A five-key sort:

```python
(overflows, unrelated, nca_size, partner_size, partner_id)
```

1. **`overflows`** — would the merge exceed `max_size`? Ranked first because the
   ceiling is a hard memory constraint and relatedness is a quality preference.
2. **`unrelated`** — do they share any ancestor at all?
3. **`nca_size`** — how specific is that shared ancestor? Smaller is better.
4. **`partner_size`** — prefer small partners, keeping the distribution tight.
5. **`partner_id`** — determinism.

> **Decision:** the original implementation merged the two globally-smallest
> groups. That keeps sizes tight but produces biologically meaningless unions —
> the real build gave `toll_like_receptor_cascades + 12 unrelated pathways`, a
> name describing one thirteenth of its contents. Since a group is a unit of
> interpretation (one `[GROUP_CLS]`, one latent contribution), a name that
> *misdescribes* it is worse than one that is obviously vague.

Passing no `hierarchy` falls back to size-pairing, so `merge_undersized` stays
usable standalone — and there is a test pinning the old behaviour as the
contrast case.

---

## 5. Two-phase orchestration — `build_hierarchical_groups` [`build.py:96`](../../src/endo_model/data/grouping/build.py:96)

```python
reactome_cut = strategy.cut(reactome, universe, REACTOME_PHASE)
residual     = universe - reactome_assigned
go_cut       = strategy.cut(go, residual, GO_PHASE)
dropped      = residual - go_assigned
```

Reactome claims first; GO only ever sees `GO minus Reactome`. Genes in neither
hierarchy are dropped — the defensible "no hierarchical annotation" exclusion.
Currently 13,746 of 29,241.

`UnionCutStrategy.cut` picks `expand_to` by phase: `reactome_groups` (default
`None`) for Reactome, `go_groups` for GO.

### Naming — [`build.py:60`](../../src/endo_model/data/grouping/build.py:60)

`_pretty_group_name` has four cases:

| Parts | Name form | Example |
|---|---|---|
| 1 | `prefix_label` | `reactome_sensory_perception` |
| 2 | `prefix_a_and_b` | `reactome_amino_acid_and_lipid_metabolism` |
| >2, informative ancestor | `prefix_ancestor_N_pathways` | `reactome_immune_system_8_pathways` |
| >2, no useful ancestor | `prefix_first_plus_N_more` | `go_urea_cycle_plus_125_more` |

The count in case 3 is not decoration: eight immune pathways are *part of* the
immune system, and sibling branches usually live in other groups, so the bare
ancestor name would overclaim.

`_informative_ancestor` [`:42`](../../src/endo_model/data/grouping/build.py:42)
rejects an ancestor covering more than `_MAX_ANCESTOR_SHARE` (0.5) of what the
hierarchy assigns **in that phase**.

> **Decision:** the denominator is the phase's assigned-gene count, not the
> universe it was cut against. GO sees the Reactome residual and covers a
> minority of it — measured against the universe, `biological_process` scores
> 28 % and would have passed the filter, keeping the useless
> `go_biological_process_39_pathways`.

`_dedupe` [`:88`](../../src/endo_model/data/grouping/build.py:88) appends a
counter on collision — the source of `reactome_metabolism_7_pathways_1`.

---

## 6. `emit_vocabulary` — [`build_vocabulary.py:486`](../../src/scripts/build_vocabulary.py:486)

Turns `{group: [genes]}` into the JSON schema.

- Genes in no group are dropped.
- Emission follows `gene_order` (the corpus list) so `global_idx` is
  deterministic and contiguous over kept genes.
- `group_idx` counts position within each group.
- `chromosome` is `""` — the schema field exists but the hierarchical path never
  populates it.

**Contract:** `global_idx` contiguous over `[0, D)`, and `group_idx` contiguous
over `[0, G_k)` per group. `ModuleGroupEmbedding` allocates `nn.Embedding(G_k,
d_token)` per group; a gap or duplicate is an index error or a silently shared
row.

---

## 7. The greedy path (legacy)

`build_vocabulary` [`:404`](../../src/scripts/build_vocabulary.py:404) — flat
set-cover: Hallmark first-match-wins, then supplementary libraries in order via
`assign_supplementary` [`:307`](../../src/scripts/build_vocabulary.py:307), then
`split_large_groups` [`:384`](../../src/scripts/build_vocabulary.py:384) chops
oversized groups into `_000`, `_001` chunks.

Still the argparse default (`--grouping greedy`), but the shell script passes
`--grouping hierarchical` explicitly. ⚠ The mismatch is a trap for anyone
invoking the script by hand.

---

## 8. Current result

```
--cut union --min-group-size 250 --max-group-size 500 --go-groups 4 --drop-unassigned
```

| | |
|---|---|
| corpus union | 29,241 genes (11 studies; wang/zhang skipped) |
| Reactome | 10,119 genes → 28 groups (239 frontier nodes before merging) |
| GO residual | 5,376 genes → 17 groups (874 before merging) |
| dropped | 13,746 (no hierarchy) |
| **D** | **15,495** |
| **K** | **45** |
| sizes | min 252, median 336, max 493 |
| names | 15 meaningful, 30 grab-bag, 0 misleading |

The GO phase takes ~3.5 min (was ~1 s) — `_pick_partner` is O(K²) in ancestor-set
intersections across 874 initial groups. It runs once, offline.

---

## 9. Initialisation differences

| Invocation | Path |
|---|---|
| `--save-hierarchy DIR` | download structure files, exit |
| `--save-gene-sets F` | download Hallmark + Enrichr, exit |
| `--use-synthetic` | `generate_synthetic_vocabulary`, no pathway data |
| `--grouping greedy` | flat set-cover; `--max-group-size` chunks |
| `--grouping hierarchical --cut tree` | ceiling only; `expand_to` unused |
| `--grouping hierarchical --cut union` | ceiling + count + floor ← **live** |
| `--max-group-size 0` | ceiling off (union only); greedy exits 1 |
| `--drop-unassigned` | genes in no group excluded from `D` |
