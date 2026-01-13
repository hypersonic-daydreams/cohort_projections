#!/usr/bin/env python3
import json
import argparse
import os
from datetime import datetime
import uuid

TRACKING_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(TRACKING_DIR, "citation_audit.json")

def load_data():
    if not os.path.exists(JSON_FILE):
        return {"citations": []}
    with open(JSON_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)

def add_citation(args):
    data = load_data()

    new_citation = {
        "id": str(uuid.uuid4()),
        "citation_key": args.key,
        "verification_method": args.method,
        "title": args.title,
        "claim_in_paper": args.claim,
        "scite_status": args.status,
        "scite_metrics": {
            "supporting": int(args.supporting),
            "mentioning": int(args.mentioning),
            "contrasting": int(args.contrasting)
        },
        "notes": args.notes,
        "last_checked": datetime.utcnow().isoformat() + "Z"
    }

    # Strict Validation
    if args.status == "Verified":
        total_metrics = int(args.supporting) + int(args.mentioning) + int(args.contrasting)
        if total_metrics == 0:
            print(f"Error: Cannot add 'Verified' citation {args.key} with 0 metrics.")
            print("  -> Please provide --supporting, --mentioning, and --contrasting.")
            print("  -> Use 'Pending' status if data is incomplete.")
            return

    data["citations"].append(new_citation)
    save_data(data)
    print(f"Added citation: {args.key} ({args.status})")

def generate_report(args):
    data = load_data()
    print(f"{'Key':<20} | {'Method':<12} | {'Status':<10} | {'Supp/Contrast':<13} | {'Claim Summary'}")
    print("-" * 95)

    for c in data["citations"]:
        metrics = c.get("scite_metrics", {})
        supp = metrics.get("supporting", 0)
        ment = metrics.get("mentioning", 0)
        cont = metrics.get("contrasting", 0)

        # Check for anomaly: Verified but all metrics 0
        total = supp + ment + cont
        is_verified = c.get("scite_status") == "Verified"

        flag = ""
        if is_verified and total == 0:
            flag = "[!] "

        claim_short = (c["claim_in_paper"][:25] + '..') if len(c["claim_in_paper"]) > 25 else c["claim_in_paper"]
        method = c.get("verification_method", "Scite")

        key_display = flag + c['citation_key']
        print(f"{key_display:<20} | {method:<12} | {c['scite_status']:<10} | {supp}/{cont:<13} | {claim_short}")

def main():
    parser = argparse.ArgumentParser(description="Manage Scite citation audit.")
    subparsers = parser.add_subparsers()

    # Add command
    parser_add = subparsers.add_parser("add", help="Add a new verified citation")
    parser_add.add_argument("--key", required=True, help="Citation key (e.g. Smith2020)")
    parser_add.add_argument("--title", required=True, help="Paper title")
    parser_add.add_argument("--claim", required=True, help="The claim in your paper")
    parser_add.add_argument("--status", required=True, choices=["Verified", "Contested", "Nuanced", "Retracted", "Pending"])
    parser_add.add_argument("--supporting", default=0, help="Count of supporting citations")
    parser_add.add_argument("--mentioning", default=0, help="Count of mentioning citations")
    parser_add.add_argument("--contrasting", default=0, help="Count of contrasting citations")
    parser_add.add_argument("--notes", default="", help="Notes")
    parser_add.add_argument("--method", default="Scite", choices=["Scite", "Web Search"], help="Verification method used")
    parser_add.set_defaults(func=add_citation)

    # Report command
    parser_report = subparsers.add_parser("report", help="Generate status report")
    parser_report.set_defaults(func=generate_report)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
