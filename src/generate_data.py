"""
Builds a synthetic-but-realistic Twitter customer-support dataset for one brand.

WHY SYNTHETIC: the real Kaggle "Customer Support on Twitter" dataset
(thoughtvector/customer-support-on-twitter) requires an authenticated Kaggle
account to download, which wasn't available in this build environment. Rather
than fabricate a claim of using real data, this script generates a dataset
that mirrors the real dataset's exact structure (same columns as twcs.csv:
tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id) and is modeled on the real, publicly-documented
tone/format of the @AmazonHelp support handle from that dataset. See
REPORT.md "What's misleading about my headline number" for the honest
limitations this creates.

Run: python src/generate_data.py
Output: data/raw_conversations.csv
"""
import csv
import random
from pathlib import Path

random.seed(42)

BRAND = "AmazonHelp"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Each intent has: customer message templates (with slots), a pool of agent
# resolution templates (the "how this brand historically resolves it"), and
# whether that scenario type should typically be escalated to a human.
# ---------------------------------------------------------------------------

ORDER_NUMS = [f"#{random.randint(100000000, 999999999)}" for _ in range(400)]
PRODUCTS = ["headphones", "phone case", "blender", "office chair", "running shoes",
            "laptop stand", "coffee maker", "backpack", "monitor", "board game",
            "water bottle", "desk lamp", "bluetooth speaker", "yoga mat", "keyboard"]
DAYS = ["2", "3", "5", "7", "10", "12"]
CITIES = ["Austin", "Denver", "Phoenix", "Columbus", "Atlanta", "Portland"]

INTENTS = {
    "order_status": {
        "escalate_rate": 0.05,
        "customer": [
            "Hi, my order {oid} hasn't shipped yet and it's been {days} days. Any update?",
            "can you tell me where my package {oid} is? tracking hasn't moved since Tuesday",
            "Order {oid} still says 'preparing for shipment' after {days} days, what's going on",
            "Hey {brand}, when will order {oid} actually ship? no updates on my end",
            "Is order {oid} lost? tracking page hasn't refreshed in {days} days",
        ],
        "agent": [
            "Hi there, sorry for the wait! I've checked order {oid} and it's currently at our fulfillment center, scheduled to ship within 24 hours. You'll get a tracking email once it's out. ^AG",
            "Thanks for flagging this. Order {oid} is delayed due to high demand but is confirmed to ship by tomorrow. We'll email tracking as soon as it moves. ^KP",
            "I checked order {oid} - it's processing and should update within 24-48 hrs. If it doesn't move by then, reply here and we'll escalate to the warehouse team. ^AG",
        ],
    },
    "delivery_issue": {
        "escalate_rate": 0.15,
        "customer": [
            "My order {oid} arrived completely damaged, the {product} is broken in the box",
            "Package {oid} says delivered but I never got it, checked with neighbors too",
            "The {product} I ordered ({oid}) showed up smashed, box was crushed",
            "Order {oid} marked delivered {days} days ago in {city} but nothing here, this is the second time",
            "Got the wrong item for order {oid} - ordered a {product}, got something else entirely",
        ],
        "agent": [
            "So sorry to hear that! For a damaged {product} on order {oid}, we can send a free replacement or full refund right away - which would you prefer? ^KP",
            "That's not okay, I apologize. I've located order {oid} - since it shows delivered but you don't have it, I'm filing a claim with the carrier. You'll see a resolution (refund or reship) within 48 hours. ^MT",
            "Ugh, that's frustrating. For the wrong item on order {oid}, I've started a free return label and a replacement order for the correct {product} is going out today. ^AG",
        ],
    },
    "refund_or_return": {
        "escalate_rate": 0.10,
        "customer": [
            "I want to return the {product} from order {oid}, it's not what I expected",
            "Requested a refund for {oid} {days} days ago and still haven't seen the money back",
            "Can I cancel order {oid}? Haven't shipped yet I think",
            "Refund for order {oid} was approved but never hit my card, it's been {days} days",
            "Return label for {oid} expired before I could print it, can you resend",
        ],
        "agent": [
            "No problem! I've started a return for the {product} on order {oid} - prepaid label is on its way to your email. Refund posts once it's scanned at drop-off. ^MT",
            "I see the refund for {oid} was processed on our end {days} days ago - refunds typically take 3-5 business days to appear on your statement. If it's been longer than that, we'll escalate to billing. ^KP",
            "Order {oid} hasn't shipped yet so I've cancelled it and refunded the full amount - you'll see it within 3-5 business days. ^AG",
            "Sent a fresh return label for {oid} to your email, this one's valid for 30 days. ^MT",
        ],
    },
    "billing_or_payment": {
        "escalate_rate": 0.35,
        "customer": [
            "I was charged twice for order {oid}, need one of these reversed ASAP",
            "There's a charge on my card I don't recognize, possibly related to order {oid}?? this looks like fraud",
            "My promo code didn't apply on order {oid}, got charged full price",
            "Charged for order {oid} but I cancelled it, why do I still see the charge",
            "Someone used my account to place order {oid}, I never authorized this purchase",
        ],
        "agent": [
            "I can see the duplicate charge on {oid} - I've flagged it for our billing team, the extra charge will be reversed within 3-5 business days. ^KP",
            "I understand the concern - I'm unable to view full card details here, but I've escalated this to our account security team who will investigate the charge on {oid} and contact you directly. ^MT",
            "Sorry about that! I've applied a courtesy credit for the promo discount that should've applied to {oid}. ^AG",
            "Pending charges for cancelled orders typically drop off within 24-48 hours and don't actually get collected - if {oid} still shows as charged after 3 days, let us know. ^KP",
        ],
    },
    "account_access": {
        "escalate_rate": 0.55,
        "customer": [
            "Locked out of my account, password reset email never arrives",
            "I think my account was hacked, orders I didn't place are showing up",
            "Can't log in, it says my account is suspended, no idea why",
            "Two-factor code never comes through, been trying for {days} minutes",
            "My email got changed on the account without me doing it",
        ],
        "agent": [
            "That sounds like it needs our account security team - I've escalated this and they'll reach out to verify your identity and secure the account. Please don't share any codes over social media. ^MT",
            "For login issues, try requesting the reset from a private/incognito window first - if it still doesn't arrive, I'm escalating to our access team now. ^KP",
            "This needs identity verification which we can't do here for security reasons - escalating you to our account recovery team, they'll DM you next steps. ^AG",
        ],
    },
    "product_or_other": {
        "escalate_rate": 0.05,
        "customer": [
            "Does the {product} come in other colors?",
            "Just wanted to say the {product} I got is great, fast shipping too!",
            "What's the warranty on the {product}?",
            "Is the {product} compatible with older models?",
            "General question - do you guys price match other retailers?",
        ],
        "agent": [
            "Glad you're loving it! Thanks for the kind words. ^AG",
            "Great question - the {product} comes with a standard 1-year warranty, details are on the product page. ^KP",
            "Color availability varies by seller, best to check the listing's variant options directly. ^MT",
            "We don't officially price match, but keep an eye out for deals during sale events. ^AG",
        ],
    },
}

NOISE_SUFFIXES = ["", "", "", " pls help", " smh", " this is ridiculous", " thanks", " 😡", " ??", " URGENT"]


def fill(template, oid, product, days, city):
    return template.format(oid=oid, brand=BRAND, product=product, days=days, city=city)


def build_rows():
    rows = []
    tweet_id = 1
    conv_meta = []  # for golden set sampling later: (customer_tweet_id, intent, should_escalate, agent_reply_id)

    per_intent_count = 45  # 6 intents * 45 = 270 conversations -> ~540 rows
    for intent, cfg in INTENTS.items():
        for i in range(per_intent_count):
            oid = random.choice(ORDER_NUMS)
            product = random.choice(PRODUCTS)
            days = random.choice(DAYS)
            city = random.choice(CITIES)

            cust_template = random.choice(cfg["customer"])
            agent_template = random.choice(cfg["agent"])
            cust_text = fill(cust_template, oid, product, days, city) + random.choice(NOISE_SUFFIXES)
            agent_text = fill(agent_template, oid, product, days, city)

            should_escalate = random.random() < cfg["escalate_rate"]

            cust_id = tweet_id
            agent_id = tweet_id + 1
            author_cust = f"cust_{random.randint(10000, 99999)}"

            rows.append({
                "tweet_id": cust_id, "author_id": author_cust, "inbound": True,
                "created_at": f"2025-0{random.randint(1,9)}-{random.randint(10,28):02d}",
                "text": cust_text, "response_tweet_id": agent_id,
                "in_response_to_tweet_id": "",
            })
            rows.append({
                "tweet_id": agent_id, "author_id": BRAND, "inbound": False,
                "created_at": f"2025-0{random.randint(1,9)}-{random.randint(10,28):02d}",
                "text": agent_text, "response_tweet_id": "",
                "in_response_to_tweet_id": cust_id,
            })
            conv_meta.append({
                "customer_tweet_id": cust_id, "intent": intent,
                "should_escalate": should_escalate, "agent_tweet_id": agent_id,
            })
            tweet_id += 2

    random.shuffle(rows)
    return rows, conv_meta


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rows, conv_meta = build_rows()

    with open(DATA_DIR / "raw_conversations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tweet_id", "author_id", "inbound", "created_at",
                                           "text", "response_tweet_id", "in_response_to_tweet_id"])
        w.writeheader()
        w.writerows(rows)

    with open(DATA_DIR / "conversation_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["customer_tweet_id", "intent", "should_escalate", "agent_tweet_id"])
        w.writeheader()
        w.writerows(conv_meta)

    print(f"Wrote {len(rows)} tweets ({len(conv_meta)} conversations) to {DATA_DIR}")


if __name__ == "__main__":
    main()
