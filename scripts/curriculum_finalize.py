from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from adaptive_v11_curriculum.hashguard import tree_sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--source',required=True); ap.add_argument('--v11',default='artifacts/v11_policy'); ap.add_argument('--hash-file',required=True); a=ap.parse_args()
    root=Path(a.root); expected=Path(a.hash_file).read_text().strip(); got=tree_sha256(a.v11)
    if got!=expected: raise RuntimeError('frozen V11 hash changed; refusing finalization')
    dst=root/'final_policy'; tmp=root/'final_policy.tmp'
    if tmp.exists(): shutil.rmtree(tmp)
    shutil.copytree(a.source,tmp)
    if dst.exists(): shutil.rmtree(dst)
    tmp.rename(dst)
    obj={'final_policy':str(dst),'source_checkpoint':a.source,'frozen_v11':'artifacts/v11_policy','frozen_v11_sha256':got,'untouched_final_test':str(root/'final_test_manifest_UNTOUCHED.json')}
    (root/'FINAL.json').write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
if __name__=='__main__': main()
