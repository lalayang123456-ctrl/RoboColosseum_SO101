#!/usr/bin/env python3
"""Read-only Router check: health, token/institution, and the Open-track models for this robot.

    colosseum-client/.venv/bin/python colosseum/tools/router_check.py colosseum-client/configs/robot.so101.yaml
"""
import json
import sys

import httpx

from colosseum_client import RobotClientConfig
from colosseum_client.evaluation import api_url

config = RobotClientConfig.from_yaml(sys.argv[1])
base = api_url(config)
headers = {'Authorization': f'Bearer {config.token}', 'X-Colosseum-Institution': config.institution}
with httpx.Client(base_url=base, timeout=30) as client:
    print('health:', client.get('/api/health').json())
    tasks = client.get('/api/eval/tasks', params={'robot_id': config.robot_type}, headers=headers)
    print(f'token + institution: HTTP {tasks.status_code}', '(OK)' if tasks.status_code == 200 else tasks.text[:200])
    if tasks.status_code == 200:
        print('fine-tuning tasks:', [task['id'] for task in tasks.json()['tasks']])
    board = client.get('/api/leaderboard', params={'robot': config.robot_type, 'track': 'open'}).json()['board']
    print(f'Open-track policies registered for {config.robot_type}:')
    for row in board:
        print(json.dumps({key: row[key] for key in ('policyId', 'modelUrl', 'revision', 'subfolder', 'status', 'evals')}))
