"""
debug_session.py — Walk through ONE session using the REAL evaluator logic
(same simulator functions as evaluator/local_evaluator.py), printing the
full back-and-forth. Run from the repo root:

    python debug_session.py
    python debug_session.py --index 5
    python debug_session.py --scenario browsing
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from starter.agent import Agent
from evaluator.local_evaluator import (
    load_jsonl, catalog_index, coarse_category, initial_message,
    customer_reply, materialize_hidden_fields, normalize_recommendations,
    MAX_TURNS, TOP_K,
)


def run_session(agent: Agent, sample: dict, catalog_ids, categories, products):
    session_id = f"debug_{uuid.uuid4().hex}"
    target = str(sample["ground_truth"]["parent_asin"])
    scenario = sample["scenario_type"]

    print("=" * 70)
    print(f"SESSION: {sample['sample_id']}   scenario={scenario}   target={target}")

    card, behavior = materialize_hidden_fields(sample, products)
    print(f"(hidden) intent_card: {card}")
    print(f"(hidden) behavior: {behavior}")
    print("=" * 70)

    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    agent.reset(session_id, sample["user_profile"])

    disclosed = set()
    boundary_used = False
    override_applied = scenario != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n--- Turn {turn} ---")
        print(f"CUSTOMER: {user_message}")

        response = agent.respond(session_id, user_message, turn, TOP_K)
        print(f"AGENT MSG: {response.get('message')}")
        print(f"ASK_ATTRIBUTE: {response.get('ask_attribute')}")

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"RECOMMENDATIONS ({len(ranked)}): {ranked}")

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print(f"\n>>> HIT at rank {rank} (turn {turn}) <<<")
            return

        if turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )

    print("\n>>> MISS — target never appeared in top 10 <<<")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--catalog", type=str, default="data/catalog.jsonl")
    parser.add_argument("--dataset", type=str, default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    if args.scenario:
        samples = [s for s in samples if s["scenario_type"] == args.scenario]
        if not samples:
            print(f"No sessions with scenario_type='{args.scenario}'")
            return

    if args.index >= len(samples):
        print(f"Only {len(samples)} sessions available; index {args.index} out of range")
        return

    agent = Agent(args.catalog)
    run_session(agent, samples[args.index], catalog_ids, categories, products)


if __name__ == "__main__":
    main()