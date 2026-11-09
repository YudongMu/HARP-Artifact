"""Stable identities for actual solver settings and noise schedules."""
import hashlib
import json


def schedule_digest(record):
    effective = dict(record,config={k:v for k,v in record['config'].items() if not k.startswith('_')})
    return hashlib.sha256(json.dumps(effective,sort_keys=True,allow_nan=False).encode()).hexdigest()
