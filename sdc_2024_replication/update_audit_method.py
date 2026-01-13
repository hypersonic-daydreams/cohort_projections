import json
import os

JSON_FILE = "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/citation_management/tracking/citation_audit.json"

def update_citations():
    if not os.path.exists(JSON_FILE):
        print("File not found")
        return

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    for citation in data.get("citations", []):
        if "verification_method" not in citation:
            citation["verification_method"] = "Web Search"

    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated {len(data['citations'])} citations.")

if __name__ == "__main__":
    update_citations()
