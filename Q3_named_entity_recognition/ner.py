"""
Q3 - Customer Feedback Analysis (Topic 13)
==========================================
This project repurposes NER into practical customer-feedback analysis.

What it does:
1. Runs lightweight entity extraction on customer feedback text
2. Extracts product names, issue types, and sentiment polarity
3. Builds aggregate analytics (top products, top issue categories)
4. Saves charts and an annotated HTML report
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt


FEEDBACKS = [
    "The camera quality on Pixel 8 is excellent, but battery drains too fast.",
    "My iPhone 14 gets hot during gaming and the screen freezes sometimes.",
    "Galaxy Buds sound great, but the left earbud disconnects frequently.",
    "Dell XPS laptop performance is smooth, however fan noise is very loud.",
    "The Sony TV picture is amazing and setup was easy.",
    "MacBook Air keyboard feels premium, but speaker crackling issue persists.",
    "OnePlus charger stopped working after two weeks.",
    "The washing machine motor makes a strange noise and vibration.",
    "This refrigerator cooling is perfect and energy usage is low.",
    "My smartwatch strap broke and heart rate sensor is inaccurate.",
    "Customer support replaced my router quickly, very satisfied.",
    "Headphone mic quality is poor and calls sound muffled.",
    "The tablet display is bright and battery backup is decent.",
    "Printer setup was confusing and wifi connection keeps dropping.",
    "Air purifier works well, but filter is expensive.",
]

PRODUCTS = {
    "pixel 8": "Phone",
    "iphone 14": "Phone",
    "galaxy buds": "Audio",
    "dell xps": "Laptop",
    "sony tv": "TV",
    "macbook air": "Laptop",
    "oneplus charger": "Accessory",
    "washing machine": "Appliance",
    "refrigerator": "Appliance",
    "smartwatch": "Wearable",
    "router": "Network",
    "headphone": "Audio",
    "tablet": "Tablet",
    "printer": "Printer",
    "air purifier": "Appliance",
}

ISSUE_PATTERNS = {
    "battery": ["battery", "drains", "backup"],
    "heating": ["hot", "heating"],
    "connectivity": ["disconnects", "wifi", "connection", "dropping"],
    "audio": ["speaker", "mic", "muffled", "sound"],
    "hardware": ["broke", "stopped working", "motor", "vibration"],
    "performance": ["freezes", "lag", "slow", "noise"],
    "price": ["expensive"],
    "setup": ["setup", "confusing"],
    "sensor": ["sensor", "inaccurate"],
}

POS_WORDS = {
    "excellent", "great", "amazing", "easy", "smooth", "premium",
    "perfect", "low", "satisfied", "bright", "decent", "works well",
}
NEG_WORDS = {
    "drains", "hot", "freezes", "disconnects", "loud", "crackling",
    "stopped", "strange", "broke", "poor", "confusing", "dropping",
    "expensive", "inaccurate", "muffled", "issue",
}


def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text.lower()).strip()


def extract_products(text: str) -> list[str]:
    norm = normalize(text)
    found = []
    for product in PRODUCTS:
        if product in norm:
            found.append(product)
    return found


def extract_issues(text: str) -> list[str]:
    norm = normalize(text)
    found = []
    for issue, keywords in ISSUE_PATTERNS.items():
        if any(word in norm for word in keywords):
            found.append(issue)
    return found


def detect_sentiment(text: str) -> str:
    norm = normalize(text)
    pos_hits = sum(1 for word in POS_WORDS if word in norm)
    neg_hits = sum(1 for word in NEG_WORDS if word in norm)

    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


def annotate_html(text: str, products: list[str], issues: list[str], sentiment: str) -> str:
    rendered = text

    for p in sorted(products, key=len, reverse=True):
        rendered = re.sub(
            re.escape(p),
            f'<mark class="product">{p} <sup>PRODUCT</sup></mark>',
            rendered,
            flags=re.IGNORECASE,
        )

    for issue in issues:
        rendered = re.sub(
            issue,
            f'<mark class="issue">{issue} <sup>ISSUE</sup></mark>',
            rendered,
            flags=re.IGNORECASE,
        )

    return f'<div class="card"><p>{rendered}</p><p><strong>Sentiment:</strong> {sentiment}</p></div>'


def main() -> None:
    rows = []
    product_counter = Counter()
    issue_counter = Counter()
    sentiment_counter = Counter()

    print("=" * 70)
    print("Q3 - CUSTOMER FEEDBACK ANALYSIS")
    print("=" * 70)

    for idx, text in enumerate(FEEDBACKS, start=1):
        products = extract_products(text)
        issues = extract_issues(text)
        sentiment = detect_sentiment(text)

        product_counter.update(products)
        issue_counter.update(issues)
        sentiment_counter.update([sentiment])

        rows.append((idx, text, products, issues, sentiment))

        print(f"\\n[{idx}] {text}")
        print(f"  Products : {products if products else ['none']}")
        print(f"  Issues   : {issues if issues else ['none']}")
        print(f"  Sentiment: {sentiment}")

    print("\\n" + "-" * 70)
    print("Summary")
    print("-" * 70)
    print(f"Total feedback entries: {len(FEEDBACKS)}")
    print(f"Products detected    : {sum(product_counter.values())}")
    print(f"Issue mentions       : {sum(issue_counter.values())}")
    print(f"Sentiment split      : {dict(sentiment_counter)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    top_products = product_counter.most_common(6)
    axes[0].bar(
        [p for p, _ in top_products],
        [c for _, c in top_products],
        color="#4c78a8",
        edgecolor="white",
    )
    axes[0].set_title("Top Products Mentioned")
    axes[0].tick_params(axis="x", rotation=35)

    top_issues = issue_counter.most_common(6)
    axes[1].bar(
        [i for i, _ in top_issues],
        [c for _, c in top_issues],
        color="#f58518",
        edgecolor="white",
    )
    axes[1].set_title("Top Issue Categories")
    axes[1].tick_params(axis="x", rotation=35)

    sentiment_order = ["positive", "neutral", "negative"]
    axes[2].bar(
        sentiment_order,
        [sentiment_counter.get(s, 0) for s in sentiment_order],
        color=["#54a24b", "#9d9d9d", "#e45756"],
        edgecolor="white",
    )
    axes[2].set_title("Sentiment Distribution")

    plt.tight_layout()
    results_path = os.path.join(os.path.dirname(__file__), "results.png")
    plt.savefig(results_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {results_path}")

    html_cards = []
    for _, text, products, issues, sentiment in rows:
        html_cards.append(annotate_html(text, products, issues, sentiment))

    html_report = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Customer Feedback Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; padding: 0 14px; background: #f9fafb; color: #222; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #555; margin-top: 6px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; margin: 12px 0; }}
    mark.product {{ background: #dbeafe; padding: 2px 5px; border-radius: 5px; }}
    mark.issue {{ background: #fee2e2; padding: 2px 5px; border-radius: 5px; }}
    sup {{ font-size: 0.65em; color: #444; }}
  </style>
</head>
<body>
  <h1>Customer Feedback Analysis</h1>
  <p class=\"meta\">Topic 13 alignment: extracting products/issues and summarizing feedback patterns.</p>
  {''.join(html_cards)}
</body>
</html>
"""

    html_path = os.path.join(os.path.dirname(__file__), "annotated_article.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"Saved: {html_path}")


if __name__ == "__main__":
    main()
