import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p))
    print(d['dtype'], 'cfg', d['cfg'], 'step', d['step_ms'], 'rtf', d['rtf_tokens'], 'mps', round(d['mem_after_gen']['mps_driver_gb'], 2), 'rms', d['rms_dbfs'])
