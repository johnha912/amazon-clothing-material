"""Task C charts for the report and the slides.

Owner: Nguyen Ha (Issue #3).

Run from the repo root:

    python analysis/task_c_charts.py

Reads the committed Task C result files (no model is retrained):

- results/review_sentiment_lexicon_check.csv — share of lexicon-positive
  material sentences at each star rating.
- results/review_sentiment_by_material.csv — mean material sentiment and
  star average per material class.

Writes two Plotly charts, each as .html and .png:

- results/task_c_positive_by_stars
- results/task_c_sentiment_by_material
"""

# %% Imports and config
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Classes backed by only a handful of products cannot carry a sentiment
# estimate; they are drawn grey and labelled with their product count.
MIN_PRODUCTS_FOR_COLOR = 30

# %% Chart 1: share of positive material sentences by star rating

lex = pd.read_csv(RESULTS_DIR / "review_sentiment_lexicon_check.csv")
lex["stars"] = lex.rating.astype(int).astype(str) + " ★"
lex["share_pct"] = (lex.share_positive * 100).round(1)
lex["label"] = (lex.share_pct.astype(str) + "%<br>("
                + lex.polar_sentences.map("{:,}".format) + " sentences)")

fig1 = px.bar(lex, x="stars", y="share_positive", text="label",
              labels={"stars": "Star rating of the review",
                      "share_positive": "Share of material sentences "
                                        "the lexicon calls positive"},
              title="Fabric praise rises with the star rating<br>"
                    "<sup>Share of material-related sentences the lexicon "
                    "scores positive, by the review's star rating</sup>")
fig1.update_traces(textposition="outside")
fig1.update_yaxes(tickformat=".0%", range=[0, 1.08])
fig1.write_html(RESULTS_DIR / "task_c_positive_by_stars.html",
                include_plotlyjs="cdn")
fig1.write_image(RESULTS_DIR / "task_c_positive_by_stars.png",
                 width=1000, height=620, scale=2)
print("saved results/task_c_positive_by_stars.html/.png")

# %% Chart 2: material sentiment by material class

bym = (pd.read_csv(RESULTS_DIR / "review_sentiment_by_material.csv")
       .sort_values("material_sentiment"))
bym["label"] = (bym.material_sentiment.round(3).astype(str) + "<br>(n="
                + bym.products.map("{:,}".format) + ")")
bym["color"] = ["#1f77b4" if n >= MIN_PRODUCTS_FOR_COLOR else "#bdbdbd"
                for n in bym.products]

# One trace with per-bar colors: splitting the bars into two traces by
# reliability would break the value ordering of the classes.
fig2 = go.Figure(go.Bar(x=bym.material_sentiment, y=bym.material,
                        orientation="h", text=bym.label,
                        textposition="outside",
                        marker_color=bym.color))
fig2.update_layout(
    title="Fabric sentiment by material class (confidence intervals "
          "pending)<br><sup>Mean per-product material sentiment; "
          "grey classes have too few products to read</sup>",
    xaxis_title="Mean material sentiment (share positive, per product)",
    yaxis_title="Material class")
fig2.update_xaxes(range=[0, 0.95])
fig2.write_html(RESULTS_DIR / "task_c_sentiment_by_material.html",
                include_plotlyjs="cdn")
fig2.write_image(RESULTS_DIR / "task_c_sentiment_by_material.png",
                 width=1000, height=620, scale=2)
print("saved results/task_c_sentiment_by_material.html/.png")
