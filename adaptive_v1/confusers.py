from v11_2.counterfactual import PREFERRED_CONFUSERS


def default_competitor(target:str,roles):
    for c in PREFERRED_CONFUSERS.get(target,()):
        if c in roles and c!=target:return c
    return next((r for r in roles if r!=target),None)
