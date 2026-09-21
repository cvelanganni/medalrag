# generate_multi_golden.py
"""
Generates 3 versions of the golden dataset
with different reference lengths to benchmark
the impact on RAGAs metrics.

Version A — Short (1-2 sentences)
Version B — Medium (3-4 sentences, current v2)
Version C — Long (6-8 sentences, expert)
"""
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

# Load current golden dataset
with open("golden_dataset_v2.json") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions")

def shorten_reference(question: str, reference: str) -> str:
    """Generates a SHORT (1-2 sentences) reference."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"""You are a medical expert. Compress this HIV clinical reference 
into EXACTLY 1-2 sentences keeping only the most critical facts 
(drug name, dose, key reason).

Question: {question}
Full reference: {reference}

Write ONLY the compressed reference (1-2 sentences, no preamble):"""}],
        temperature=0, max_tokens=100
    )
    return resp.choices[0].message.content.strip()

def extend_reference(question: str, reference: str) -> str:
    """Generates a LONG (6-8 sentences) reference."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"""You are a senior HIV clinician. Extend this reference into
a comprehensive 6-8 sentence clinical answer covering:
- Exact drug names and doses
- Mechanism/reason for the recommendation  
- Monitoring requirements
- Alternative options if applicable
- Source guideline (NIH/HHS or GESIDA)

Question: {question}
Current reference: {reference}

Write ONLY the extended reference (6-8 sentences):"""}],
        temperature=0, max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# Generate 3 versions
dataset_short  = []
dataset_medium = []  # current v2
dataset_long   = []

print("\nGenerating short and long references...")
for i, item in enumerate(dataset):
    print(f"  [{i+1}/{len(dataset)}] {item['question'][:50]}...")

    q   = item["question"]
    ref = item["reference"]

    # Short version
    short = shorten_reference(q, ref)
    dataset_short.append({
        "category" : item["category"],
        "lang"     : item["lang"],
        "question" : q,
        "reference": short,
    })

    # Medium = keep current reference
    dataset_medium.append(item)

    # Long version
    long = extend_reference(q, ref)
    dataset_long.append({
        "category" : item["category"],
        "lang"     : item["lang"],
        "question" : q,
        "reference": long,
    })

# Save
with open("golden_dataset_short.json",  "w", encoding="utf-8") as f:
    json.dump(dataset_short,  f, indent=2, ensure_ascii=False)
with open("golden_dataset_medium.json", "w", encoding="utf-8") as f:
    json.dump(dataset_medium, f, indent=2, ensure_ascii=False)
with open("golden_dataset_long.json",   "w", encoding="utf-8") as f:
    json.dump(dataset_long,   f, indent=2, ensure_ascii=False)

# Show samples
print("\n=== Sample SHORT reference ===")
print(dataset_short[8]["reference"])
print("\n=== Sample MEDIUM reference ===")
print(dataset_medium[8]["reference"])
print("\n=== Sample LONG reference ===")
print(dataset_long[8]["reference"])

# Stats
import statistics
short_lens  = [len(d["reference"].split()) for d in dataset_short]
medium_lens = [len(d["reference"].split()) for d in dataset_medium]
long_lens   = [len(d["reference"].split()) for d in dataset_long]

print(f"\n=== Reference length stats ===")
print(f"Short  : {statistics.mean(short_lens):.0f} words avg")
print(f"Medium : {statistics.mean(medium_lens):.0f} words avg")
print(f"Long   : {statistics.mean(long_lens):.0f} words avg")
print("\nDone! Now run:")
print("  uv run evaluate.py --versions v9_hybrid --sleep 15")
print("  (change GOLDEN_DATASET_PATH for each version)")