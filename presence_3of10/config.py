POOL = (
    "code",
    "writing",
    "math",
    "research",
    "design",
    "operations",
    "translation",
    "financial_analysis",
    "database_design",
    "customer_support_triage",
)

# Ground truth of the target MAS.
# This is used only by:
#   1. the target MAS executor, because these agents actually exist
#   2. evaluation AFTER belief prediction
#
# It is NEVER passed to the probe policy or belief model.
HIDDEN = (
    "translation",
    "financial_analysis",
    "database_design",
)

SESSION_ID = "presence_3of10_easy_001"
