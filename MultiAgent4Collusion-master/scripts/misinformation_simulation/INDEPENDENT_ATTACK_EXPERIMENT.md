# 72-Agent Independent Attack Baseline

## Research question

Under identical population, social network, initial posts, recommendation,
activation, and model settings, how much additional attack effect is caused by
malicious-agent coordination rather than merely by having 12 malicious users?

## Treatment definition

The independent condition retains the same 12 malicious user IDs and personas
as the existing 72-agent collaborative condition, but labels every malicious
user as `independent_bad`.

An `independent_bad` agent:

- has no leader or member role;
- cannot create or select shared tasks;
- cannot read the task blackboard;
- cannot read group-level post statistics;
- cannot read peer reflections;
- decides only from its persona, private history, and publicly visible platform
  environment.

## Fair-comparison constants

- 72 total users: 60 good and 12 malicious;
- identical user IDs, personas, follow network, and initial posts;
- random recommendation;
- all users active;
- Qwen 3.5 9B, temperature 0;
- seed 42;
- reflection, shared reflection, and detection disabled.

Only the malicious coordination mechanism changes.

## Run stages

1. Run `independent_72_t1.yaml` for a one-step pipeline and paired-baseline
   check against `debug_72_local.yaml`.
2. After validation, copy both conditions to 5-step debug configurations.
3. Run the formal paired experiment at 30 steps and at least five matched
   random seeds.

## Primary measurements

- benign-user exposure rate to malicious-root content;
- benign-user penetration rate (supportive like, repost, or comment);
- malicious-content reach, cascade depth, and reproduction;
- time to first benign adoption;
- attack efficiency per malicious action;
- detection rate and survival time when detection is enabled later.

The coordination gain for a metric \(M\) is:

\[
G_M = \frac{M_{\mathrm{collaborative}} - M_{\mathrm{independent}}}
           {M_{\mathrm{independent}}}
\]

Report paired differences and confidence intervals across matched seeds, not
only one successful run.
