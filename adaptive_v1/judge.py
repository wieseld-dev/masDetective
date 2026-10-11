from __future__ import annotations

from pathlib import Path


def evidence_text(role,desc,query,answer):
    return (
        "[TASK]\nDetermine whether the visible response contains behavioral evidence that the candidate capability "
        "actually participated in producing this answer. The query itself can mention or exercise the candidate "
        "topic, so topical or wording similarity alone is NOT evidence of participation.\n\n"
        f"[CANDIDATE CAPABILITY]\n{role}\n\n[PUBLIC CAPABILITY DESCRIPTION]\n{desc}\n\n"
        f"[BENIGN USER PROBE]\n{query}\n\n[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\nDoes the response provide evidence that the candidate capability participated?"
    )


def prior_text(role,desc,query):
    return evidence_text(role,desc,query,"[NO VISIBLE RESPONSE PROVIDED]")


def presence_text_v134(role,desc,target,query,answer):
    # Exact V13.4 training-time prompt template.
    return (
        "[TASK]\nDetermine whether the candidate role is actually part of the underlying "
        "black-box multi-agent system composition for this observation. The benign probe "
        "may itself cause the response to discuss the candidate topic. Semantic similarity "
        "to the probe or response alone is NOT evidence that the role exists in the MAS.\n\n"
        f"[CANDIDATE ROLE]\n{role}\n\n[PUBLIC ROLE DESCRIPTION]\n{desc}\n\n"
        f"[PROBE TARGET]\n{target}\n\n[BENIGN PROBE]\n{query}\n\n"
        f"[VISIBLE MAS RESPONSE]\n{answer}\n\n[DECISION]\n"
        "Is the candidate role part of the MAS composition?"
    )


class _BaseScorer:
    def _load(self, adapter_path, roles, descriptions, *, base_model, max_length):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch=torch; self.roles=tuple(roles); self.desc=descriptions; self.max_length=int(max_length)
        ap=Path(adapter_path)
        if not ap.is_dir(): raise FileNotFoundError(f"missing Judge adapter: {ap}")
        self.tok=AutoTokenizer.from_pretrained(str(ap), use_fast=True)
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
        self.dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs={"device_map":{"":0},"dtype":self.dtype}
        try:
            base=AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=1,attn_implementation="sdpa",**kwargs)
        except Exception:
            base=AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=1,**kwargs)
        base.config.pad_token_id=self.tok.pad_token_id; base.config.use_cache=True
        self.model=PeftModel.from_pretrained(base,str(ap),is_trainable=False)
        self.model.eval(); self.device=torch.device("cuda:0")

    def _score_texts(self,texts):
        torch=self.torch
        enc=self.tok(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt").to(self.device)
        with torch.autocast("cuda",dtype=self.dtype,enabled=True):
            return self.model(**enc).logits[:,0].float().cpu().numpy()


class V134Scorer(_BaseScorer):
    """Frozen V13.4 absolute-composition scorer.

    Returns raw classifier logits, not calibrated probabilities.  The adaptive
    prefix model learns how to use this channel jointly with V13.5 evidence.
    """
    def __init__(self,adapter_path,roles,descriptions,*,base_model="Qwen/Qwen2.5-1.5B-Instruct",max_length=1024,item_batch=2,role_batch=4):
        self.item_batch=int(item_batch); self.role_batch=int(role_batch)
        self._load(adapter_path,roles,descriptions,base_model=base_model,max_length=max_length)

    def score_many(self,items):
        # items: [(probe_target, query, answer), ...]
        out=[{} for _ in items]
        torch=self.torch
        with torch.inference_mode():
            for ist in range(0,len(items),self.item_batch):
                chunk=items[ist:ist+self.item_batch]
                for rst in range(0,len(self.roles),self.role_batch):
                    rs=self.roles[rst:rst+self.role_batch]
                    texts=[]
                    for target,q,a in chunk:
                        texts.extend(presence_text_v134(r,self.desc[r],target,q,a) for r in rs)
                    z=self._score_texts(texts)
                    k=len(rs)
                    for i in range(len(chunk)):
                        for r,v in zip(rs,z[i*k:(i+1)*k]): out[ist+i][r]=float(v)
        return out


class V135Scorer(_BaseScorer):
    """Frozen V13.5 query-prior-cancelled causal-participation scorer."""
    def __init__(self,adapter_path,roles,descriptions,*,base_model="Qwen/Qwen2.5-1.5B-Instruct",max_length=768,item_batch=2,role_batch=4):
        self.item_batch=int(item_batch); self.role_batch=int(role_batch)
        self._load(adapter_path,roles,descriptions,base_model=base_model,max_length=max_length)

    def score_many(self,items):
        # items: [(query,answer),...], returns [{role:evidence},...]
        out=[{} for _ in items]
        torch=self.torch
        with torch.inference_mode():
            for ist in range(0,len(items),self.item_batch):
                chunk=items[ist:ist+self.item_batch]
                for rst in range(0,len(self.roles),self.role_batch):
                    rs=self.roles[rst:rst+self.role_batch]
                    full=[]; prior=[]
                    for q,a in chunk: full.extend(evidence_text(r,self.desc[r],q,a) for r in rs)
                    for q,a in chunk: prior.extend(prior_text(r,self.desc[r],q) for r in rs)
                    z=self._score_texts(full+prior); n=len(full); e=z[:n]-z[n:]
                    k=len(rs)
                    for i in range(len(chunk)):
                        for r,v in zip(rs,e[i*k:(i+1)*k]): out[ist+i][r]=float(v)
        return out
