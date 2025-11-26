import os
import json


filenames = set(os.listdir('videos'))

content = json.load(open('WLASL_v0.3.json'))

missing_ids = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' not in filenames:
            missing_ids.append(video_id)


with open('missing.txt', 'w') as f:
    f.write('\n'.join(missing_ids))

