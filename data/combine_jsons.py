import sys
path=sys.argv[1]

import os
import json

if __name__ == "__main__":
    combined_scores = {}
    for f in os.listdir(path):
        if not f.endswith(".json") or f.startswith("."):
            continue
        combined_scores.update(json.load(open(os.path.join(path, f))))

    json.dump(combined_scores, open(sys.argv[2], 'w'))