# Multi-seed evaluation and robustness

Held-out users: 604. Agents: 5 training seeds each; intervals are 95% t-intervals across seeds. Baselines are deterministic (random uses a fixed seed) and are evaluated once. Every policy sees the same users and evaluation seed.

## Nominal simulator

| Policy | Return | Session length | Watch | CTR | ILD | Repeat |
|---|---|---|---|---|---|---|
| SlateQ | 10.14 [9.79, 10.50] | 13.13 | 4.41 | 0.447 | 0.726 | 0.526 |
| PPO (Plackett-Luce) | 9.32 [9.01, 9.63] | 11.74 | 3.97 | 0.478 | 0.629 | 0.544 |
| DQN | 8.74 [8.41, 9.06] | 11.86 | 3.62 | 0.422 | 0.647 | 0.383 |
| MMR | 8.51 | 11.20 | 3.78 | 0.486 | 0.619 | 0.750 |
| greedy-click | 8.45 | 11.09 | 3.75 | 0.491 | 0.585 | 0.753 |
| retrieval | 8.37 | 11.12 | 3.67 | 0.487 | 0.579 | 0.741 |
| PPO (proto-actions) | 8.36 [8.22, 8.50] | 11.10 | 3.66 | 0.485 | 0.592 | 0.732 |
| random | 4.89 | 10.68 | 1.85 | 0.250 | 0.683 | 0.234 |
| popularity | 4.05 | 10.90 | 2.16 | 0.255 | 0.738 | 0.838 |

![nominal returns](nominal_returns.png)

## Robustness (return gain over greedy-click)

| Scenario | DQN | SlateQ | PPO (Plackett-Luce) | PPO (proto-actions) |
|---|---|---|---|---|
| nominal | +0.28 [-0.04, +0.61] | +1.69 [+1.34, +2.04] | +0.86 [+0.56, +1.17] | -0.10 [-0.23, +0.04] |
| fatigue_x0.5 | -0.88 [-1.40, -0.35] | +0.81 [+0.17, +1.44] | +0.88 [+0.41, +1.34] | +0.22 [-0.09, +0.52] |
| fatigue_x1.5 | +1.01 [+0.82, +1.20] | +2.02 [+1.88, +2.16] | +1.00 [+0.74, +1.27] | +0.15 [+0.05, +0.26] |
| easier_clicks | +2.13 [+1.73, +2.53] | +5.64 [+4.90, +6.39] | +1.78 [+1.30, +2.27] | +0.17 [-0.12, +0.47] |
| pickier_users | +0.01 [-0.13, +0.15] | +0.29 [+0.15, +0.43] | +0.40 [+0.19, +0.61] | -0.04 [-0.20, +0.11] |
| patient_users | +1.06 [+0.65, +1.47] | +3.09 [+2.54, +3.63] | +1.65 [+1.36, +1.94] | +0.40 [+0.34, +0.46] |
| impatient_users | +0.30 [+0.12, +0.48] | +1.02 [+0.94, +1.10] | +0.67 [+0.46, +0.87] | +0.00 [-0.10, +0.10] |
| sim_seed2 | +0.99 [+0.91, +1.08] | +2.03 [+1.62, +2.44] | +1.30 [+1.10, +1.51] | +0.53 [+0.40, +0.65] |

Policy ranking by mean return per scenario:

- **nominal**: SlateQ > PPO (Plackett-Luce) > DQN > MMR > greedy-click > retrieval > PPO (proto-actions) > random > popularity
- **fatigue_x0.5**: PPO (Plackett-Luce) > SlateQ > retrieval > PPO (proto-actions) > MMR > greedy-click > DQN > random > popularity
- **fatigue_x1.5**: SlateQ > DQN > PPO (Plackett-Luce) > MMR > PPO (proto-actions) > greedy-click > retrieval > random > popularity
- **easier_clicks**: SlateQ > DQN > PPO (Plackett-Luce) > MMR > PPO (proto-actions) > greedy-click > retrieval > random > popularity
- **pickier_users**: PPO (Plackett-Luce) > SlateQ > DQN > greedy-click > PPO (proto-actions) > MMR > retrieval > random > popularity
- **patient_users**: SlateQ > PPO (Plackett-Luce) > DQN > MMR > PPO (proto-actions) > retrieval > greedy-click > random > popularity
- **impatient_users**: SlateQ > PPO (Plackett-Luce) > DQN > MMR > PPO (proto-actions) > greedy-click > retrieval > random > popularity
- **sim_seed2**: SlateQ > PPO (Plackett-Luce) > DQN > retrieval > PPO (proto-actions) > MMR > greedy-click > random > popularity

![robustness](robustness.png)
