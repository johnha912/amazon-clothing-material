# Discussion and limitations

Three limitations bound what these results can claim.

| Limitation | Why it matters | Response |
|---|---|---|
| The label is a proxy | "Dominant fiber" is clear for a 90/10 shirt, weaker for blends where a minor fiber drives the feel, and misleading for shoes whose string describes the sole | Decide whether shoes stay in scope before the final sample is frozen |
| Labeled listings are not a random subset | Only 46% carry a composition string, and brands that publish one may write different copy | Task A scores may not transfer to the unlabeled majority; check the gap on the full sample |
| Average rating is a coarse target | It mixes product quality with sizing, expectations, and delivery | The weak Task B result shows that listings carry little rating signal, not that buyers ignore material |

The Task A model is best used as a search and accessibility aid: it recovers a field sellers already state in prose. Its predictions should be shown with their confidence and never stored as seller-provided facts, especially for regulated claims such as leather or wool content.

Next, we will rerun every model on the 250,000-product sample with the merged class list from `src/labels.py`, complete the DistilBERT comparison, and add a confidence display to the demo app so that its predictions show the same uncertainty described here.
