# Discussion and limitations

Four limitations bound what these results can claim.

| Limitation | Why it matters | Response |
|---|---|---|
| The label is a proxy | "Dominant fiber" is clear for a 90/10 shirt, weaker for blends where a minor fiber drives the feel, and misleading for shoes whose string describes the sole | Decide whether shoes stay in scope before the final sample is frozen |
| Labeled listings are not a random subset | Only 36.4% of the ~250k sample (46% of the pilot) carry a composition string, and brands that publish one may write different copy | Task A scores may not transfer to the unlabeled majority; check the gap on the full sample |
| The shared cleaning step missed some composition strings | `src/labels.strip_composition` left a percentage in 13,373 of 90,979 labeled listings, which inflated Task A macro-F1 by about 0.06 to 0.07; `analysis/load_data.py` now blanks every clause holding a digit followed by a percent sign, which empties the text of 405 labeled listings that we keep; 42 labeled listings (10 in the test set) still contain a stray percent sign in forms the labeling rule does not read, such as "%100 cotton" or "lOO%", and 1,048 spell out "N percent" | Scores use the cleaned text; a stricter cleaner would need a rerun and is left for later, and the shared helper in `src/` should be fixed so every script is clean by default |
| Average rating is a coarse target | It mixes product quality with sizing, expectations, delivery, and the noise of small rating counts | The weak Task B result shows that listings carry little rating signal, not that buyers ignore material |

The Task A model is best used as a search and accessibility aid: it recovers a field sellers already state in prose. Its predictions should be shown with their confidence and never stored as seller-provided facts, especially for regulated claims such as leather or wool content.
