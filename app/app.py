"""TrustPanel prototype: browse personas, talk to them, test a concept, and see how far to trust them.

    streamlit run app/app.py
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

from trustpanel import llm as llm_module  # noqa: E402
from trustpanel.data import STARS, Product  # noqa: E402
from trustpanel.llm import ContentFiltered  # noqa: E402
from trustpanel.panel import ask_persona, panel_distribution  # noqa: E402
from trustpanel.personas import persona_from_json  # noqa: E402
from trustpanel.validation import expected  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
RESULTS = os.path.join(ROOT, "results", "main")
TOPIC_LABELS = {"baby_care": "Baby care", "bath_tissue": "Bath & facial tissue", "feminine_care": "Feminine care",
                "kitchen_paper": "Kitchen paper & napkins", "skin_care": "Skin care", "wound_care": "Wound care"}
CHAT_RULES = ("\n\nYou are talking to an innovation team. Answer in the first person, in your own voice, "
              "in a few sentences. Only draw on the situation, needs and experiences above; if asked about "
              "something they do not cover, say you are not sure rather than inventing details.")

st.set_page_config(page_title="TrustPanel", page_icon="🧪", layout="wide")


@st.cache_data
def load_personas():
    # The local copy keeps the review quotes behind each persona; the published copy does not.
    for path in (os.path.join(ROOT, "data", "processed", "personas.json"), os.path.join(ROOT, "results", "personas.json")):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return []


@st.cache_data
def load_results():
    def read(name):
        path = os.path.join(RESULTS, name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f] if name.endswith(".jsonl") else json.load(f)
    return read("summary.json"), read("calibration.json"), read("predictions.jsonl")


@st.cache_resource
def get_llm():
    return llm_module.from_env(os.path.join(ROOT, "data", "cache"))


def quote(text: str, limit: int = 220) -> str:
    text = re.sub(r"\[\[VIDEOID:[^\]]*\]\]", "", text)  # Amazon video placeholders
    return text[:limit] + ("…" if len(text) > limit else "")


def ask_or_none(llm, persona, sample, product):
    try:
        return ask_persona(llm, persona, product, sample)
    except ContentFiltered:
        return None


def stars(dist):
    return pd.DataFrame({"share": dist}, index=[f"{s}★" for s in STARS])


personas_raw = load_personas()
personas = [persona_from_json(p) for p in personas_raw]
# Two segments can get the same persona name; the segment's key terms tell them apart.
_names = [(p.topic, p.name) for p in personas]
for p in personas:
    if _names.count((p.topic, p.name)) > 1:
        p.name = f"{p.name} ({', '.join(p.segment['top_terms'][:2])})"
summary, calibration, predictions = load_results()
priors = {r["topic"]: r["topic_prior"] for r in predictions or []}

with st.sidebar:
    st.markdown("## 🧪 TrustPanel")
    st.caption("Synthetic consumer personas built from real reviews, with a validation of how far to trust them.")
    topic = st.selectbox("Product category", sorted({p.topic for p in personas}), format_func=lambda t: TOPIC_LABELS.get(t, t))
    panel = sorted([p for p in personas if p.topic == topic], key=lambda p: -p.weight)
    if calibration:
        st.warning("Validation found that persona star ratings carry **no reliable signal** for real ratings "
                   f"(calibration slope {calibration['persona_panel']['slope']:.2f}). Use them to explore needs "
                   "and objections, not to forecast scores.")

tab_lib, tab_chat, tab_concept, tab_val = st.tabs(["Persona library", "Talk to a persona", "Concept test", "Validation"])

with tab_lib:
    st.subheader(f"{TOPIC_LABELS.get(topic, topic)}: {len(panel)} personas")
    for p in panel:
        quotes = {e["id"]: e for e in p.segment.get("evidence", [])}
        with st.expander(f"**{p.name}** · {p.weight:.0%} of reviewers", expanded=p is panel[0]):
            st.write(p.summary)
            st.caption(f"Situation: {p.situation}")
            col1, col2 = st.columns(2)
            for col, key, title in ((col1, "needs", "Needs"), (col2, "pain_points", "Pain points")):
                with col:
                    st.markdown(f"**{title}**")
                    for item in getattr(p, key):
                        st.markdown(f"- {item['need']}")
                        for rid in item["evidence"]:
                            if rid in quotes:
                                st.caption(f"↳ {rid} ({quotes[rid]['rating']}★): “{quote(quotes[rid]['text'])}”")
            st.markdown("**Deal breakers:** " + "; ".join(p.deal_breakers))
            st.caption(f"Price sensitivity: {p.price_sensitivity} · Rating tendency: {p.rating_tendency}")
            counts = p.segment.get("rating_counts", {})
            if counts:
                total = sum(counts.values()) or 1
                st.caption(f"Segment: {p.segment['n_reviews']:,} reviews · real ratings "
                           + " ".join(f"{s}★ {counts[str(s)] / total:.0%}" for s in STARS)
                           + f" · key terms: {', '.join(p.segment['top_terms'][:6])}")

with tab_chat:
    who = st.selectbox("Persona", panel, format_func=lambda p: f"{p.name} ({p.weight:.0%})")
    key = f"chat-{who.persona_id}"
    history = st.session_state.setdefault(key, [])
    for m in history:
        st.chat_message(m["role"]).write(m["content"])
    if question := st.chat_input(f"Ask {who.name} something, e.g. what would make you switch brands?"):
        history.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        try:
            reply = get_llm().chat(who.card() + CHAT_RULES, history)
        except ContentFiltered:
            reply = "_(The Azure content filter blocked this exchange. Try rephrasing.)_"
        history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

with tab_concept:
    st.write("Describe a product concept the way a shop would. Each persona rates it three times; the panel "
             "mixes their ratings by the share of consumers each persona speaks for.")
    concept = st.text_area("Concept", height=150, placeholder="Ultra-thin bamboo paper towels, 2-ply, plastic-free packaging, 6 rolls")
    price = st.number_input("Price ($)", min_value=0.0, value=9.99, step=0.5)
    if st.button("Run panel", type="primary", disabled=not concept.strip()):
        lines = concept.strip().splitlines()
        product = Product(product_id="concept", topic=topic, title=lines[0], brand="", price=price,
                          features=[], description=" ".join(lines[1:]))
        jobs = [(p, s) for p in panel for s in range(3)]
        with st.spinner(f"Asking {len(panel)} personas, 3 times each…"):
            with ThreadPoolExecutor(max_workers=9) as pool:
                answers = [a for a in pool.map(lambda job: ask_or_none(get_llm(), *job, product), jobs) if a]
        if not answers:
            st.error("No answers: every call was blocked by the content filter.")
        else:
            dist = panel_distribution(answers, panel)
            col1, col2 = st.columns([1, 2])
            col1.metric("Panel mean", f"{expected(dist):.2f}★")
            if topic in priors:
                col1.metric("Category average (real reviews)", f"{expected(priors[topic]):.2f}★")
            col1.metric("Would buy", f"{sum(a.would_buy for a in answers) / len(answers):.0%}")
            col2.bar_chart(stars(dist))
            names = {p.persona_id: p.name for p in panel}
            st.dataframe(pd.DataFrame([{"persona": names[a.source], "rating": a.rating, "would buy": a.would_buy,
                                        "reason": a.reason} for a in answers]), width="stretch", hide_index=True)
            st.info("Read the reasons, not the score: in validation the panel's stars did not track real ratings.")

with tab_val:
    if not summary:
        st.info("Run scripts/run_validation.py first.")
    else:
        st.write(f"{summary['n_products']} held-out products, brands hidden. Each predictor is compared with the "
                 "real rating distribution of the product's reviews; 95% bootstrap intervals in brackets.")
        labels = {"persona_panel": "Persona panel", "persona_calibrated": "Persona, calibrated", "generic_llm": "Generic LLM",
                  "generic_calibrated": "Generic, calibrated", "topic_prior": "Category prior"}
        metrics = [("mean_abs_error", "Mean rating error (★, lower is better)"),
                   ("wasserstein", "Distribution distance (★, lower)"),
                   ("pairwise_accuracy", "Picks the better product (higher)"),
                   ("spearman", "Rank correlation (higher)"),
                   ("bias", "Bias, predicted − real (★, 0 is best)"),
                   ("spread_ratio", "Spread vs real (1 is best)")]
        rows = []
        for metric, label in metrics:
            row = {"metric": label}
            for k in summary["predictors"]:
                v, ci = summary[k]["metrics"][metric], summary[k]["ci95"].get(metric)
                row[labels[k]] = f"{v:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{v:.2f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("**Mean rating error by category**")
        st.bar_chart(pd.DataFrame(summary["by_topic"]).T.rename(columns=labels, index=TOPIC_LABELS), stack=False)
        if predictions:
            st.markdown("**Predicted vs real mean rating, per product**")
            st.scatter_chart(pd.DataFrame([{"real mean": expected(r["truth"]), "persona panel": expected(r["persona_panel"]),
                                            "category": TOPIC_LABELS.get(r["topic"], r["topic"])} for r in predictions]),
                             x="real mean", y="persona panel", color="category")
        if calibration:
            st.caption("Calibration fitted on 60 separate products (real mean = a + b × predicted mean): "
                       + "; ".join(f"{labels[k]}: b = {v['slope']:.2f}" for k, v in calibration.items())
                       + ". A slope near 0 means the predictor's scores carry no information about real ratings.")
