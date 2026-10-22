# Donor and sample overlap between studies

### garcia_alonso_2021_a  vs  garcia_alonso_2021_b

- cells: 100,307 and 10,729 (10.7%)
- **source_sample_id: 15 shared** of 15 and 15
- patient_id: 0 shared of 15 and 15
- **Samples match and donor ids do not: the same people are carrying two identifiers, so the splitter treated them as different donors.**

**Shared samples split differently: 7 of 15.**

| sample | in garcia_alonso_2021_a | in garcia_alonso_2021_b |
|---|---|---|
| `A30` | train 17,000 | test 376 |
| `E3` | train 3,018 | val 795 |
| `SAMN15049042` | val 1,958 | train 995 |
| `SAMN15049043` | val 3,069 | train 990 |
| `SAMN15049046` | test 2,945 | val 997 |
| `SAMN15049049` | test 6,737 | train 874 |
| `SAMN15049051` | train 2,025 | test 996 |

Held-out cells belonging to a donor that also appears in the other study: **17,873**. Those cells are not cross-donor with respect to that person.
