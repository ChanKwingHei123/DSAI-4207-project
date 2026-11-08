# Distractor & Prompt-Input Build Report

## 1. Configuration & Summary
- **Master seed**: `4207`
- **Draw seeds**: `[0, 1, 2]`
- **Hard-negative pool size (`k`)**: `20` (from top-30 dense candidates)
- **Length-match tolerance**: `+/- 20%` (primary: `Qwen3-8B` tokenizer; secondary check: `Llama-3.1-8B-Instruct` tokenizer)
- **Questions processed**: `590` (`main=500`, `pilot=50`, `reserve=40`)
- **Questions with `< 20` non-rejected hard candidates**: `24`
- **Failed random length-matching slots**: `23`

## 2. Answer-Leak Rejection Counts
Official SQuAD v2.0 normalization is applied to every candidate passage and every gold answer; a passage is rejected if any normalized gold answer appears as a substring.

| Metric | Value |
| --- | --- |
| Questions with >= 1 rejected candidate in top-30 | 207 / 590 |
| Total top-30 hard candidates rejected (across 590 questions) | 958 / 17700 |
| Top-30 rejections per question (mean / median / min / max) | 1.6237 / 0.0 / 0 / 30 |
| Rejections encountered before reaching k=20 hard pool (sum / max) | 897 / 30 |
| Non-gold-article dev pool rejections per question (mean / max) | 12.9712 / 1150 |

### Distribution of Top-30 Hard-Candidate Rejections per Question
| Rejected Count (out of 30) | Remaining Non-Rejected | Number of Questions |
| --- | --- | --- |
| 0 | 30 | 383 |
| 1 | 29 | 84 |
| 2 | 28 | 31 |
| 3 | 27 | 18 |
| 4 | 26 | 10 |
| 5 | 25 | 16 |
| 6 | 24 | 8 |
| 7 | 23 | 4 |
| 8 | 22 | 4 |
| 9 | 21 | 3 |
| 10 | 20 | 5 |
| 11 | 19 | 5 |
| 12 | 18 | 1 |
| 14 | 16 | 1 |
| 15 | 15 | 3 |
| 18 | 12 | 1 |
| 20 | 10 | 1 |
| 21 | 9 | 3 |
| 22 | 8 | 1 |
| 23 | 7 | 2 |
| 24 | 6 | 2 |
| 25 | 5 | 1 |
| 27 | 3 | 1 |
| 28 | 2 | 1 |
| 30 | 0 | 1 |

## 3. Questions with Fewer Than 20 Hard-Pool Candidates
| Question ID | Split | Non-Rejected | Rejected (of 30) | Normalized Gold Answers | Question Text |
| --- | --- | --- | --- | --- | --- |
| `56e1b754cd28a01900c67abc` | `main` | 19 | 11 | `complexity`, `complexity resources` | Time and space are both examples of what type of resource? |
| `56e1c4fce3433e1400423151` | `main` | 0 | 30 | `p` | What is the expression used to represent a complexity class of counting problems? |
| `5705e63175f01819005e7720` | `main` | 6 | 24 | `california` | The cities of Los Angeles and San Diego are a part of which state? |
| `5705eccb52bb8914006896ba` | `main` | 7 | 23 | `desert`, `southern` | Which region of California is Palm Springs located in? |
| `57094d489928a8140047150e` | `main` | 10 | 20 | `pvr`, `sky` | What is the name brand of the personal video recorder that BSkyB offers? |
| `5711119cb654c5140001fae4` | `main` | 19 | 11 | `protestant` | What general religious belief did the nations that received Huguenot refugees have in common? |
| `57112686b654c5140001fbd5` | `main` | 3 | 27 | `steam` | In the Rankine cycle, what does water turn into when heated? |
| `571cac5d5efbb31900334dd4` | `main` | 19 | 11 | `earth` | Before the formation of which planet, did Sol lose oxygen 16? |
| `571cd5b1dd7acb1400e4c17c` | `main` | 5 | 25 | `gaseous oxygen`, `oxygen` | What does  oxygen therapy promote the body to take up? |
| `5726975c708984140094cb21` | `main` | 15 | 15 | `court of justice`, `italian constitutional court` | Which court argued that the Treaty of Rome did not prevent energy nationalism? |
| `572647d0708984140094c14c` | `main` | 9 | 21 | `ctenophores`, `diploblastic` | Ctenophores and cnidarians are classified as what? |
| `57268a37f1498d1400e8e340` | `main` | 12 | 18 | `ctenophores` | Turtles and jellyfish can eat large quantities of what? |
| `57281c594b864d19001644a6` | `main` | 9 | 21 | `appeal to constitutional defects`, `civil disobedience`, `nonviolent`, `violence` | Debates about civil disobedience include or exclude what major practice?  |
| `57274d9bf1498d1400e8f5f7` | `main` | 16 | 14 | `chinese` | What language is used in Chinese primary schools in Malaysia? |
| `57297781af94a219006aa4a7` | `main` | 7 | 23 | `1`, `eliminate all multiples of 1`, `eliminate all multiples of 1 that is all other numbers and produce as output only single number 1`, `only single number 1` | If 1 were to be considered as prime what would the sieve of Eratosthenes yield for all other numbers? |
| `572ff932a23a5019007fcbd6` | `main` | 2 | 28 | `islam` | What do supporters of Islamism believe their views reflect? |
| `5730b7ce069b5314008322c6` | `main` | 15 | 15 | `war`, `war erupted` | What caused the US public to condemn the occupation of the philippines? |
| `57337ea24776f41900660bd0` | `main` | 9 | 21 | `city`, `demographically` | What was Warsaw the most diverse of in Poland? |
| `573786b51c4567190057448f` | `main` | 19 | 11 | `mass`, `mass of attracting body` | What is gravitational acceleration proportional to? |
| `56e1ec83cd28a01900c67c0b` | `pilot` | 8 | 22 | `k` | The integer factorization problem essentially seeks to determine if the value of of an input is less than what variable?  |
| `571c7abfdd7acb1400e4c0be` | `pilot` | 19 | 11 | `air` | Of what did da Vinci think a part  was consumed during combustion? |
| `57287f6a3acd2414000dfa54` | `pilot` | 18 | 12 | `qu` | What form of poetry was developed in the Yuan? |
| `57300f8504bcaa1900d770d3` | `pilot` | 15 | 15 | `islamist` | Up until 1990, Saudi Arabia played an important role in restraining what groups? |
| `572656e4dd62a815002e81fe` | `reserve` | 6 | 24 | `court of justice` | Within the EU, which court believes they have the final word deciding on EU's competence? |

## 4. Failed Random Control Slots
| Question ID | Split | Seed | Slot | Hard Paragraph ID | Target Qwen3 Length |
| --- | --- | --- | --- | --- | --- |
| `5705e26d75f01819005e76d6` | `main` | 0 | 0 | `[2, 37]` | 43 |
| `5705e26d75f01819005e76d6` | `main` | 1 | 4 | `[2, 33]` | 43 |
| `5705e26d75f01819005e76d6` | `main` | 2 | 1 | `[2, 37]` | 43 |
| `5705e3f252bb89140068966e` | `main` | 0 | 1 | `[2, 33]` | 43 |
| `5705e3f252bb89140068966e` | `main` | 2 | 0 | `[2, 33]` | 43 |
| `5705fd8475f01819005e7844` | `main` | 2 | 6 | `[2, 3]` | 42 |
| `5706074552bb8914006897d8` | `main` | 0 | 4 | `[2, 33]` | 43 |
| `5706074552bb8914006897d8` | `main` | 1 | 4 | `[2, 33]` | 43 |
| `570607f575f01819005e78b7` | `main` | 0 | 7 | `[2, 33]` | 43 |
| `5706094b52bb8914006897df` | `main` | 0 | 2 | `[2, 33]` | 43 |
| `5706094b52bb8914006897df` | `main` | 1 | 1 | `[2, 37]` | 43 |
| `5706094b52bb8914006897df` | `main` | 2 | 7 | `[2, 33]` | 43 |
| `57060a6e52bb8914006897f8` | `main` | 1 | 6 | `[2, 33]` | 43 |
| `57060a6e52bb8914006897f8` | `main` | 2 | 3 | `[2, 33]` | 43 |
| `5706155352bb891400689895` | `main` | 1 | 6 | `[2, 37]` | 43 |
| `5706155352bb891400689895` | `main` | 2 | 3 | `[2, 37]` | 43 |
| `5726a1e5dd62a815002e8b86` | `main` | 0 | 7 | `[9, 31]` | 642 |
| `5705f09e75f01819005e77a5` | `pilot` | 1 | 6 | `[2, 3]` | 42 |
| `5705f09e75f01819005e77a5` | `pilot` | 2 | 6 | `[2, 3]` | 42 |
| `57269cc3dd62a815002e8b15` | `pilot` | 0 | 2 | `[9, 38]` | 788 |
| `57269cc3dd62a815002e8b15` | `pilot` | 2 | 5 | `[9, 38]` | 788 |
| `5705eb3375f01819005e7767` | `reserve` | 0 | 1 | `[2, 33]` | 43 |
| `5726b929f1498d1400e8e8e8` | `reserve` | 0 | 0 | `[9, 38]` | 788 |

## 5. Token-Length Matching Statistics
Evaluated across `13400` matched `(hard, random)` distractor pairs (`590 questions x 3 seeds x 8 slots = 14,160 pairs` when complete).

| Metric | Primary: Qwen3-8B Tokenizer | Secondary Check: Llama-3.1-8B Tokenizer |
| --- | --- | --- |
| Mean hard-negative token length | 171.6470 | 167.1107 |
| Mean random-control token length | 167.1825 | 162.6804 |
| Mean signed difference (`random - hard`) | -4.4645 | -4.4303 |
| SD of signed difference (`random - hard`) | 21.7863 | 22.7102 |
| Median signed difference (`random - hard`) | -3.0 | -3.0 |
| Mean absolute difference (`|random - hard|`) | 17.4090 | 17.5736 |
| Max absolute difference (`|random - hard|`) | 146 | 154 |
| Mean relative difference (`|random - hard| / hard`) | 9.97% | 10.26% |
| Max relative difference (`|random - hard| / hard`) | 20.00% | 44.16% |
| Mean signed diff (`main` split) | -4.5400 | -4.5013 |
| Mean signed diff (`pilot` split) | -4.5718 | -4.7481 |
| Mean signed diff (`reserve` split) | -3.4033 | -3.1793 |

## 6. Generated Output Files
| Split & Seed | Path | Row Count | SHA-256 |
| --- | --- | --- | --- |
| `main_seed0` | `data/inputs/main_seed0.jsonl` | 475 | `3277e9d236b9ba65a0e3e2e7d55398c9c66c28adf1f97390b076f75631ad5087` |
| `main_seed1` | `data/inputs/main_seed1.jsonl` | 476 | `604824d4395b25e74e9c9918940738598789aafad31c6c80a8115378ad664d62` |
| `main_seed2` | `data/inputs/main_seed2.jsonl` | 475 | `919ec2af9352486c3ddcf92a043760bfddfe5693602625fa1340c7604eb6dad5` |
| `pilot_seed0` | `data/inputs/pilot_seed0.jsonl` | 45 | `03b6a2a980b80cb207ae3abc657884e1dbe766bea7fae78d4b0873b4739dc79a` |
| `pilot_seed1` | `data/inputs/pilot_seed1.jsonl` | 45 | `d732e27e6011e6bdf3d9b5fa5c52b6c2c88424009a3ecd4322bf60162a226a89` |
| `pilot_seed2` | `data/inputs/pilot_seed2.jsonl` | 44 | `6cd3e6a738698af4d4629d35e1783a08582709a1c26fa87998231d97564cb791` |
| `reserve_seed0` | `data/inputs/reserve_seed0.jsonl` | 37 | `62c0aa919a535ea4f2544b41aa14f44c5676a266d5fe89fe92359584a6059e73` |
| `reserve_seed1` | `data/inputs/reserve_seed1.jsonl` | 39 | `782c400cd88d43aebe1eb0559270158ce3d3b5f3c0e8aeac81cabc42c7d35928` |
| `reserve_seed2` | `data/inputs/reserve_seed2.jsonl` | 39 | `80564a17aa5608cf6997f3415fc04751fd74547e0a018a9805996bc34a49a7e2` |
