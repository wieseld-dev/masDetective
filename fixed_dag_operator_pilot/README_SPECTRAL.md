# Fixed-DAG Spectral Operator Pilot

This pilot restricts the target family to five fixed 4-agent DAGs and uses V11 only as a benign probe bank.

## Core operator
For each candidate DAG G, learn a path polynomial F_G from shared per-agent matrices W_i:

    F_G = sum_{p in source->sink paths(G)} product_{i in p} W_i

For two DAG hypotheses g,h:

    K_gh = F_g - F_h

At adaptive round t with posterior p_t(G):

    C_t = sum_{g<h} p_t(g)p_t(h) K_gh^T K_gh

The continuous unit-norm query direction maximizing predicted topology separation is the principal eigenvector v1(C_t). The largest eigenvalue lambda1(C_t) is the maximum quadratic separation obtainable in that latent space.

Because deployment uses a finite V11 probe bank, the actual selector chooses the unused probe embedding q maximizing:

    q^T C_t q

This is compared against:
1. normalized predictive-disagreement selection;
2. random probing;
3. a graph-specific direct linear-map upper baseline.

## Run
Copy all files to ~/mas_rl_detective, then:

    chmod +x submit_dag_operator_poc.sh
    ./submit_dag_operator_poc.sh

The first job collects/caches 60 V11 probes across all 5 fixed DAGs. The dependent GPU job fits the models and evaluates spectral adaptive probing.

## Send back
Send the blocks:

    FIXED-DAG OPERATOR PILOT
    ACTIVE IDENTIFICATION — PATH OPERATOR / SPECTRAL EIGEN POLICY
    ACTIVE IDENTIFICATION — DIRECT GRAPH MAP / SPECTRAL UPPER BASELINE
    INITIAL PRINCIPAL EIGENMODE — PATH OPERATOR
    INTERPRETATION GATE

## Interpretation
- Direct spectral fails -> fixed topology is not sufficiently observable under this setup.
- Direct works, path operator fails -> the shared matrix/path algebra is wrong or too restrictive.
- Path operator works, spectral ~= random -> representation contains structure but acquisition rule is weak.
- Spectral > disagreement > random -> the eigen-based active probing mechanism is doing useful work.
