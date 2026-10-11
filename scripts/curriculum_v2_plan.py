from adaptive_v11_curriculum_v2.config import STAGES, planned_train_candidate_executions

for s in STAGES:
    print('\t'.join(map(str,[
        s.stage, s.horizon, s.train_systems_per_cycle, s.dev_systems, s.cycles,
        s.group_size, s.lr, s.kl_beta, s.retention_weight, s.mc_weight, s.name,
    ])))
