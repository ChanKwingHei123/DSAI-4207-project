# P7 Frozen Main Split Report (Decision D12 + P7 Delta)

## 1. Freeze Policy & Two-Pass Replacement Rule
- **Exclusion rule (Decision D12)**: Exclude every question in `data/sampling/main.json` that is either:
  1. Missing from any of `data/inputs/main_seed{0,1,2}.jsonl` (P2c incomplete-seed exclusion due to failed random length-match slots), or
  2. Listed in `data/audit/exclusions.json` with `adjudicated_flag == 1` (blind audit exclusion).
- **Eligible reserve pool**: Questions in `data/sampling/reserve.json` present in all three `data/inputs/reserve_seed{0,1,2}.jsonl` files and not audit-excluded.
- **Two-pass replacement rule**:
  - **Pass 1 (Same-article replacements)**: Iterate over all excluded/incomplete main questions in `main.json` order; if an unused eligible reserve question from the same `article_index` exists, assign the first such reserve question in `reserve.json` order.
  - **Pass 2 (Cross-article fallbacks)**: Iterate over remaining unassigned excluded/incomplete main questions in `main.json` order; assign each to the next unused eligible reserve question in `reserve.json` order.

## 2. Summary Counts
| Metric | Count |
| --- | ---: |
| **Final frozen `main` N** | **500** |
| Original `main.json` questions | 500 |
| Retained original `main.json` questions | 482 |
| Total excluded & replaced questions | 18 |
| Incomplete-seed exclusions (missing >=1 seed) | 10 |
| Audit exclusions (`adjudicated_flag == 1`) | 8 |
| Overlap (incomplete-seed AND audit exclusion) | 0 |
| Reserve pool total (`reserve.json`) | 40 |
| Eligible reserve questions (complete in all 3 seeds) | 39 |
| Ineligible reserve questions | 1 |
| Same-article replacements (Pass 1) | 8 |
| Cross-article fallbacks (Pass 2) | 10 |
| Unused eligible reserve remaining | 21 |

### Ineligible Reserve Questions
| reserve_qid | article | reason |
| --- | --- | --- |
| `5705eb3375f01819005e7767` | 2 (Southern_California) | incomplete_seeds (missing: 0) |

## 3. Complete Replacement Table
| old_qid | old_article | reason | new_qid | new_article | same_article |
| --- | --- | --- | --- | --- | --- |
| `5705e26d75f01819005e76d6` | 2 (Southern_California) | incomplete_seeds (missing: 0,1,2) | `56de3f784396321400ee26fa` | 0 (Normans) | false |
| `5705e3f252bb89140068966e` | 2 (Southern_California) | incomplete_seeds (missing: 0,2) | `56e181d9e3433e1400422fa3` | 1 (Computational_complexity_theory) | false |
| `5705e63175f01819005e7720` | 2 (Southern_California) | incomplete_seeds (missing: 0) | `57094ca7efce8f15003a7dd7` | 3 (Sky_(United_Kingdom)) | false |
| `5705fd8475f01819005e7844` | 2 (Southern_California) | incomplete_seeds (missing: 2) | `570d2f5bfed7b91900d45cd2` | 4 (Victoria_(Australia)) | false |
| `5706074552bb8914006897d8` | 2 (Southern_California) | incomplete_seeds (missing: 0,1) | `571163172419e314009555ea` | 6 (Steam_engine) | false |
| `570607f575f01819005e78b7` | 2 (Southern_California) | incomplete_seeds (missing: 0) | `571cb010dd7acb1400e4c12b` | 7 (Oxygen) | false |
| `5706094b52bb8914006897df` | 2 (Southern_California) | incomplete_seeds (missing: 0,1,2) | `571ce3745efbb31900334e24` | 7 (Oxygen) | false |
| `57060a6e52bb8914006897f8` | 2 (Southern_California) | incomplete_seeds (missing: 1,2) | `572650325951b619008f6fa9` | 8 (1973_oil_crisis) | false |
| `5706155352bb891400689895` | 2 (Southern_California) | incomplete_seeds (missing: 1,2) | `5726b929f1498d1400e8e8e8` | 9 (European_Union_law) | false |
| `57108c95b654c5140001f97c` | 5 (Huguenot) | audit_exclusion | `571112ada58dae1900cd6bce` | 5 (Huguenot) | true |
| `5726a1e5dd62a815002e8b86` | 9 (European_Union_law) | incomplete_seeds (missing: 0) | `572656e4dd62a815002e81fe` | 9 (European_Union_law) | true |
| `572746addd62a815002e9a61` | 19 (Private_school) | audit_exclusion | `572750df5951b619008f8830` | 19 (Private_school) | true |
| `57274971708984140094dbbb` | 19 (Private_school) | audit_exclusion | `5726722bdd62a815002e852a` | 10 (Amazon_rainforest) | false |
| `572a1a5c6aef051400155287` | 22 (Economic_inequality) | audit_exclusion | `5729e02f1d0469140077963c` | 22 (Economic_inequality) | true |
| `572a20816aef0514001552e7` | 22 (Economic_inequality) | audit_exclusion | `5727fc37ff5b5019007d9a19` | 22 (Economic_inequality) | true |
| `572982e66aef051400154f92` | 27 (Prime_number) | audit_exclusion | `572985011d04691400779502` | 27 (Prime_number) | true |
| `572ffc99947a6a140053cef6` | 30 (Islamism) | audit_exclusion | `57302faa04bcaa1900d77311` | 30 (Islamism) | true |
| `573098f38ab72b1400f9c5d2` | 31 (Imperialism) | audit_exclusion | `57308cf88ab72b1400f9c57a` | 31 (Imperialism) | true |

## 4. Per-Article Question Counts (Before vs After Replacement)
| article_index | article_title | before | excluded | added | after | delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Normans | 8 | 0 | 1 | 9 | +1 |
| 1 | Computational_complexity_theory | 17 | 0 | 1 | 18 | +1 |
| 2 | Southern_California | 15 | 9 | 0 | 6 | -9 |
| 3 | Sky_(United_Kingdom) | 9 | 0 | 1 | 10 | +1 |
| 4 | Victoria_(Australia) | 10 | 0 | 1 | 11 | +1 |
| 5 | Huguenot | 18 | 1 | 1 | 18 | 0 |
| 6 | Steam_engine | 18 | 0 | 1 | 19 | +1 |
| 7 | Oxygen | 20 | 0 | 2 | 22 | +2 |
| 8 | 1973_oil_crisis | 9 | 0 | 1 | 10 | +1 |
| 9 | European_Union_law | 20 | 1 | 2 | 21 | +1 |
| 10 | Amazon_rainforest | 15 | 0 | 1 | 16 | +1 |
| 11 | Ctenophora | 14 | 0 | 0 | 14 | 0 |
| 12 | Fresno,_California | 12 | 0 | 0 | 12 | 0 |
| 13 | Packet_switching | 9 | 0 | 0 | 9 | 0 |
| 14 | Black_Death | 9 | 0 | 0 | 9 | 0 |
| 15 | Geology | 10 | 0 | 0 | 10 | 0 |
| 16 | Pharmacy | 11 | 0 | 0 | 11 | 0 |
| 17 | Civil_disobedience | 17 | 0 | 0 | 17 | 0 |
| 18 | Construction | 8 | 0 | 0 | 8 | 0 |
| 19 | Private_school | 10 | 2 | 1 | 9 | -1 |
| 20 | Harvard_University | 9 | 0 | 0 | 9 | 0 |
| 21 | Jacksonville,_Florida | 8 | 0 | 0 | 8 | 0 |
| 22 | Economic_inequality | 25 | 2 | 2 | 25 | 0 |
| 23 | University_of_Chicago | 15 | 0 | 0 | 15 | 0 |
| 24 | Yuan_dynasty | 19 | 0 | 0 | 19 | 0 |
| 25 | Immune_system | 18 | 0 | 0 | 18 | 0 |
| 26 | Intergovernmental_Panel_on_Climate_Change | 9 | 0 | 0 | 9 | 0 |
| 27 | Prime_number | 13 | 1 | 1 | 13 | 0 |
| 28 | Rhine | 24 | 0 | 0 | 24 | 0 |
| 29 | Scottish_Parliament | 16 | 0 | 0 | 16 | 0 |
| 30 | Islamism | 17 | 1 | 1 | 17 | 0 |
| 31 | Imperialism | 16 | 1 | 1 | 16 | 0 |
| 32 | Warsaw | 20 | 0 | 0 | 20 | 0 |
| 33 | French_and_Indian_War | 15 | 0 | 0 | 15 | 0 |
| 34 | Force | 17 | 0 | 0 | 17 | 0 |
| **Total** | **All 35 articles** | **500** | **18** | **18** | **500** | **+0** |

## 5. Input & Output SHA-256 Digests
| Artifact | Path | Count | SHA-256 |
| --- | --- | ---: | --- |
| Raw SQuAD 2.0 dev | `data/raw/dev-v2.0.json` | - | `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` |
| Sampling manifest | `data/sampling/manifest.json` | - | `98dab46790100b14c6dc00234c10b85e54df702c951bcedc6b5ab7a3a3c9dcb5` |
| Audit exclusions | `data/audit/exclusions.json` | 8 | `b04f425f16c826ea05ad490ea4270dadca88348b6d4796dde5bd59eb2d3acdb1` |
| Frozen `main` | `data/frozen/main.json` | 500 | `db28adc0e11ad8227c3746748ee142675900f07d83dc2ef1710af6af58a931d3` |
| Frozen `main_seed0` | `data/frozen/main_seed0.jsonl` | 500 | `20d0e47919895e6629cc00aad2e4b4107ba8a51380ba26d44926a8da5096d539` |
| Frozen `main_seed1` | `data/frozen/main_seed1.jsonl` | 500 | `f37a62b6d3914cc0e325d1a8172d2d179d1cc73f4e9d3e61fdb485c472eeebb2` |
| Frozen `main_seed2` | `data/frozen/main_seed2.jsonl` | 500 | `6a30a3e9c6f558f2f22e3deb871972cc416c1a1cfae0c613949ed9d58eeea667` |
