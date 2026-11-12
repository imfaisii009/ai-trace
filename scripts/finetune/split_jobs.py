# Pick edit sources from commons.json and split them into one job file per agent.
import json, sys, urllib.parse
src, out, n_src, n_parts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
P = ["Recreate this exact photo.",
     "Enhance this photo while keeping everything the same.",
     "Same photo, same scene and framing, but sharper and cleaner, as if taken with a better camera.",
     "Same photo, same scene and framing, but with warmer, golden late-afternoon light.",
     "Same photo, same scene and framing, but on an overcast day with soft even light.",
     "Same photo, same scene and framing, but remove small distracting objects.",
     "Same photo, same scene and framing, but with slightly more vivid colours.",
     "Same photo, same scene and framing, but tidy and clean everything.",
     "Same photo, same scene and framing, but early evening with the lights on.",
     "Same photo, same scene and framing, but add one small object that fits naturally.",
     "Same photo, same scene and framing, but change the colour of the main object.",
     "Same photo, but make it look professionally edited.",
     "Same photo, same scene and framing, but after light rain, with wet surfaces.",
     "Same photo, same scene and framing, but replace the background details with similar ones."]
import os; os.makedirs(out, exist_ok=True)
rows = [o for o in json.load(open(src)) if o.get('thumb')][:n_src]
jobs = []
for i, o in enumerate(rows):
    u = urllib.parse.urlsplit(o['thumb'])
    jobs.append({'sha1': o['sha1'], 'url': f'https://upload.wikimedia.org{u.path}', 'prompt': P[i % len(P)]})
for k in range(n_parts):
    json.dump(jobs[k::n_parts], open(f'{out}/part{k}.json', 'w'), indent=0)
print(len(jobs), 'jobs in', n_parts, 'parts')
