"""CPU-only smoke test for calibration transformations, no BGU data."""
import collections
import math
from analysis.role_direct_within_v4.v512_calibration_audit import (validation_calibrators,calibrate,METHODS)
roles=['role_a','role_b','role_c']
systems={}
for i in range(18):
    sysid=f'fake_{i}'
    systems[sysid]=({},set([roles[i%3]]))
scores={(sid,r): float(5*(r=='role_b')+(.8 if r in pos else 0)+i%4) for i,(sid,(_,pos)) in enumerate(systems.items()) for r in roles}
cal=validation_calibrators(scores,systems,roles)
assert all(cal['validation_coverage'][r]['val_present']==6 for r in roles)
for method in METHODS:
    assert all(math.isfinite(calibrate(r,scores[(sid,r)],method,cal)) for sid in systems for r in roles)
assert cal['calibration_uses_test'] is False
print('V512_CPU_TESTS_PASS')
