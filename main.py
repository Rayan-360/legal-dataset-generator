import groq
import json
import time
import re
import os
import random
import hashlib

# ✅ Initialize Groq client using environment variable
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

# ✅ Updated Prompt Function (matches your CSV structure)
def get_prompt():
    seed = random.randint(10000, 99999)
    return f"""
You are a legal case generation assistant. Your job is to generate *synthetic employment law disputes* involving termination, retaliation, or discrimination.

Each output must include:
- A *3–5 sentence realistic summary* involving workplace termination, retaliation, or discrimination.
- A list of *3 lists of legal factors* (each 3–6 items long):
  - First list: current case factors
  - Second list: factors from a *plaintiff-side precedent case*
  - Third list: factors from a *defendant-side precedent case*
- A *3-ply legal argumentation*:
  - ply_1: plaintiff's initial argument
  - ply_2: defendant's counterargument
  - ply_3: plaintiff's rebuttal

Use the following fixed list of 20 legal factors. For each list, select 3–6 relevant ones. Each item should follow the format "F# (P)" or "F# (D)".

Factor List:
F1 – At-Will Employment Clause (D)  
F2 – Employment Contract Clauses (P)  
F3 – Protected Class Membership (P)  
F4 – Discriminatory Motive Evidence (P)  
F5 – Equal Treatment of Peers (D)  
F6 – Documented Performance Issues (D)  
F7 – Retaliation Evidence (P)  
F8 – Whistleblower Activity (P)  
F9 – Job Description Compliance (D)  
F10 – Failure to Accommodate (P)  
F11 – Disparate Impact Evidence (P)  
F12 – Prior Discipline History (D)  
F13 – Breach of Company Policy (D)  
F14 – Constructive Discharge Allegation (P)  
F15 – Harassment Complaint Filed (P)  
F16 – Internal Investigation Outcome (D)  
F17 – Temporal Proximity to Termination (P)  
F18 – Legitimate Business Reason (D)  
F19 – Witness Corroboration (P)  
F20 – Employee Handbook Violations (D)

Output Format:
{{
  "summary": "Realistic summary of the case.",
  "instruction": [
    ["F# (P)", "F# (D)", ...],    // current case factors
    ["F# (P)", ...],             // plaintiff precedent
    ["F# (D)", ...]              // defendant precedent
  ],
  "argument_ply": {{
    "ply_1": "The plaintiff argues that...",
    "ply_2": "The defendant counters that...",
    "ply_3": "The plaintiff rebuts that..."
  }}
}}

Return only the JSON object. Output must begin with '{{' and end with '}}'. Do not include any explanation. Randomization ID: {seed}
"""

# ✅ Hashing function to detect duplicate samples
def hash_sample(sample_json):
    return hashlib.md5(json.dumps(sample_json, sort_keys=True).encode()).hexdigest()

# ✅ Core function to generate samples
def generate_samples(batch_size=200, outfile="employment_dataset.jsonl"):
    seen = set()

    # Load existing data to avoid duplicates
    if os.path.exists(outfile):
        with open(outfile, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    seen.add(hash_sample(existing))
                except:
                    continue

    with open(outfile, "a", encoding="utf-8") as f:
        for i in range(batch_size):
            print(f"🔄 Generating sample {i + 1} of {batch_size}...")
            retries = 3
            while retries > 0:
                try:
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful legal assistant."},
                            {"role": "user", "content": get_prompt()}
                        ],
                        temperature=0.9
                    )
                    content = response.choices[0].message.content.strip()

                    # Extract JSON from output
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        json_text = match.group(0)
                        try:
                            sample = json.loads(json_text)
                            sample_hash = hash_sample(sample)
                            if sample_hash not in seen:
                                seen.add(sample_hash)
                                f.write(json.dumps(sample) + "\n")
                                print("✅ Sample saved.")
                                break
                            else:
                                print("⚠ Duplicate sample. Skipping.")
                                break
                        except json.JSONDecodeError:
                            print("❌ Invalid JSON. Skipping.")
                            break
                    else:
                        print("⚠ No JSON found in output.")
                        break

                except Exception as e:
                    print(f"⚠ Error: {e}. Retrying...")
                    retries -= 1
                    time.sleep(2)

# 🔧 Run generator
generate_samples(batch_size=200)  # You can adjust batch_size as needed