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


def prior_text(role,desc,query):return evidence_text(role,desc,query,"[NO VISIBLE RESPONSE PROVIDED]")


class V135Scorer:
    def __init__(self,adapter_path,roles,descriptions,*,base_model="Qwen/Qwen2.5-1.5B-Instruct",max_length=768,item_batch=4):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        self.torch=torch;self.roles=tuple(roles);self.desc=descriptions;self.max_length=int(max_length);self.item_batch=int(item_batch)
        ap=Path(adapter_path)
        if not ap.is_dir():raise FileNotFoundError(f"missing V13.5 adapter: {ap}")
        self.tok=AutoTokenizer.from_pretrained(str(ap),use_fast=True)
        if self.tok.pad_token is None:self.tok.pad_token=self.tok.eos_token
        self.dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs={"device_map":{"":0},"dtype":self.dtype}
        try:base=AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=1,attn_implementation="sdpa",**kwargs)
        except Exception:base=AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=1,**kwargs)
        base.config.pad_token_id=self.tok.pad_token_id;base.config.use_cache=True
        self.model=PeftModel.from_pretrained(base,str(ap),is_trainable=False);self.model.eval();self.device=torch.device("cuda:0")
    def score_many(self,items):
        # items: [(query,answer),...], returns [{role:evidence},...]
        out=[];torch=self.torch
        with torch.inference_mode():
            for st in range(0,len(items),self.item_batch):
                chunk=items[st:st+self.item_batch]; full=[];prior=[]
                for q,a in chunk:
                    for r in self.roles:full.append(evidence_text(r,self.desc[r],q,a))
                for q,a in chunk:
                    for r in self.roles:prior.append(prior_text(r,self.desc[r],q))
                texts=full+prior;enc=self.tok(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt").to(self.device)
                with torch.autocast("cuda",dtype=self.dtype,enabled=True):z=self.model(**enc).logits[:,0].float()
                n=len(full);e=(z[:n]-z[n:]).cpu().numpy();
                for i in range(len(chunk)):
                    vals=e[i*len(self.roles):(i+1)*len(self.roles)];out.append({r:float(v) for r,v in zip(self.roles,vals)})
        return out
