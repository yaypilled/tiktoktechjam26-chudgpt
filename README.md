TechJam 2026 — Conversational E-Commerce Search Agent (Track #4) by Chudgpt

Project Overview

This project is our submission for Track #4: E-Commerce AI Conversational Search Result Recommendation at TikTok TechJam 2026.

The challenge: build an AI shopping agent that, through a multi-turn conversation, figures out what a customer is looking for and returns the correct hidden target product — ranked as early and as highly as possible — within a maximum of 10 turns, across four customer behavior types (Buying, Browsing, Intent Override, Boundary).

Our solution is a stateful BM25 retrieval agent built on SQLite's FTS5 full-text search engine. Rather than treating each turn as an isolated search query, the agent accumulates information across the whole conversation, actively asks clarifying questions to draw out more detail from the customer, and filters results against extracted budget constraints.

Final result on the 200-session public set:
Metric	Weak baseline (provided) vs	Our final agent
Hit Rate@10	12.5% vs 84.5%
MRR	0.068 vs 0.515
MTTC	9.81 vs 4.57
Technical Score	0.107	0.705
This is a ~6.5x improvement over the provided weak starter, achieved with a fully rule-based pipeline — no external LLM calls, no API costs, and no dependency that could fail or rate-limit during judging.

---
How Our Solution Addresses the Problem Statement
The problem statement asks for an agent that turns a vague request into a useful search plan through gradual, multi-turn disclosure (category → use case → material → style → budget), and that handles four distinct customer behaviors. Our design maps directly onto this:
Gradual disclosure → state accumulation. Every message's search-relevant terms are added to a running list rather than replacing the previous turn's context, so early signals (e.g. a stated material) are never lost by later turns.
"A better question can be more valuable than another retrieval call" → active clarification. The agent tracks which attributes it has already asked about and always asks about the next most useful one (`use_case → feature → material → color → style → size → budget`) via the `ask_attribute` field. This was the single highest-leverage design decision: asking questions is what unlocks the evaluator's simulated customer to reveal specific hard/soft constraints from its hidden intent card.
Buying (hard constraint early) → budget extraction and filtering. A regex detects numeric budget statements and filters retrieved candidates against a price lookup table, without penalizing products with missing price data.
Boundary (no clear preference) → graceful fallback. When a customer has nothing to add, the agent simply moves on to the next attribute in its list rather than getting stuck.
Intent Override (customer corrects themselves) → handled by accumulation design. Because terms accumulate rather than overwrite, and because the underlying BM25 search is an OR across all known terms, a correction like "actually, white sneakers" adds new signal without requiring explicit contradiction-detection logic — this was also tested with LLM-based rewriting (see Limitations).
---
Development Tools Used
VS Code — primary editor for all Python development
Windows Terminal / PowerShell — running the evaluator, debug scripts, and Git commands
Git & GitHub — version control and team collaboration
Python 3.13
APIs Used
Google Gemini API (`gemini-3.6-flash`, via the `google-genai` SDK) — evaluated for intent-override query rewriting and considered for semantic reranking. Not included in the final shipped agent — see Limitations for why.
Libraries and Frameworks Used
`sqlite3` (Python standard library) — powers our retrieval engine via its built-in FTS5 full-text search module with BM25 ranking
`re` (standard library) — tokenization, stopword filtering, budget/price extraction
`json`, `pathlib` (standard library) — catalog loading and file handling
`google-genai` — used only in the experimental Gemini query-rewriting branch (not in the final submitted agent)
`sentence-transformers` — used only in an experimental hybrid dense-retrieval branch (not in the final submitted agent)
Our final submitted `agent.py` has zero external dependencies beyond the Python standard library.
Datasets and Assets Used
Amazon Reviews 2023 (McAuley Lab, UCSD), `Clothing_Shoes_and_Jewelry` category — the source of both the frozen 50,000-product catalog and the session data, as provided by the organizers via the official Clothing 5-core leave-last-out split.
Organizer-provided 200 labeled public development sessions (`data/public_set.jsonl`) used for local testing and iteration.
No manually labelled or externally sourced data was added.
---
Setup and Installation Instructions
Clone this repository:
```bash
   git clone https://github.com/yaypilled/tiktoktechjam26-chudgpt.git
   cd tiktoktechjam26-chudgpt
   ```
Download and extract the catalog (if not already present in `data/`):
```bash
   gzip -dk catalog.jsonl.gz
   mv catalog.jsonl data/catalog.jsonl
   ```
Verify against the published `SHA256SUMS` if desired.
Requirements: Python 3.10+. The final submitted agent (`starter/agent.py`) uses only the Python standard library — no `pip install` required to run it.
(Optional, only if exploring the experimental branches described below: `pip install google-genai sentence-transformers`, and set a `GEMINI_API_KEY` environment variable — never commit this key.)
---
Steps to Reproduce Our Results
From the repository root, run the local evaluator:
```bash
   python -m evaluator.local_evaluator
   ```
This loads the 50,000-product catalog, runs all 200 public sessions through `starter/agent.py`, and writes aggregate metrics to `results.json`.
Expected output: Hit Rate@10 ≈ 0.845, MRR ≈ 0.515, MTTC ≈ 4.57, Technical Score ≈ 0.705.
To inspect a single conversation turn-by-turn (useful for debugging or demoing):
```bash
   python debug_session.py --scenario browsing
   python debug_session.py --scenario intent_override
   python debug_session.py --index 0
   ```
This prints the full customer/agent exchange for one session, including hit/miss and rank.
---
Limitations and What We'd Improve With More Time
Current limitations:
Retrieval is purely keyword-based (BM25); it cannot match semantically related but lexically different terms (e.g. "cold, wet trail runs" vs. a product described as "waterproof hiking gear").
The clarification order is fixed across all product categories, rather than adapting to what's actually most discriminating for a given product type.
Intent Override handling relies on term accumulation rather than explicit correction-detection; it works reasonably well but doesn't actively suppress now-outdated constraints.
What we tried and deliberately did not ship, with reasons:
LLM-based query rewriting on intent overrides (Gemini) — produced correct, sensible edits, but measured zero net improvement in Hit Rate@10, MRR, or MTTC across the full 200-session evaluation. It also hit Google's free-tier quota (20 requests/day) well before completing a single full evaluation run, making it operationally infeasible at the 800-session private evaluation scale without a paid plan.
User-profile preference-tag seeding — seeding search terms from historical `preference_tags` (e.g. "comfort," "fit") measurably regressed Hit Rate@10 (0.845 → 0.825) because these tags are too generic and diluted otherwise-precise queries.
Category-adaptive clarification ordering — showed no clear net benefit (0.845 → 0.835, within noise) once tested against the full session set; most of its intended benefit was already captured by our fixed ordering.
Hybrid BM25 + dense embedding retrieval — using `sentence-transformers` embeddings combined via reciprocal rank fusion, this notably regressed MRR (0.515 → 0.281) because title-only embeddings were a weaker signal than our tuned BM25 search, and equal-weight fusion dragged correct BM25 rankings down.
Given more time, we would:
Build embeddings over richer product text (not just titles) and weight the BM25/dense fusion based on measured per-signal reliability rather than equal weighting, before re-attempting hybrid retrieval.
Derive clarification ordering empirically from the actual distribution of hard constraints per product category in the training data, rather than guessing category buckets by hand.
Explore a paid or self-hosted LLM option to revisit semantic reranking and override handling without the free-tier quota ceiling that blocked us this time.
---
Team Member Contributions
Lim Qin Hui - Set up the repository, facilitated implementation trials of the LLM API
Hein Min Htet - Key provider of ideas on how to make the BM25 algorithm more efficient, consolidated files for submission
Law Jun Hao - Coordinated the testing of our agent, assisted in the design and build of the code
Tian Kai Yang - Designed and built the core retrieval agent (agent.py)

