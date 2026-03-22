Your code anneals the eps_pos and eps_vel thresholds based on train_progress:

eps_pos = eps_pos_start + train_progress * (eps_pos_end - eps_pos_start)
mid_done is triggered when:

So as training progresses:

Progress	eps_pos	eps_vel	Meaning
0%	0.20m	0.50 m/s	Easy — "within 20cm and 0.5 m/s"
50%	0.125m	0.30 m/s	Medium
91%	0.063m	0.14 m/s	Hard — "within 6.3cm and 0.14 m/s"
100%	0.05m	0.10 m/s	Final — "within 5cm and 0.1 m/s"
At 50% progress, the agent was hitting mid_done 50%+ of the time because "get within 12.5cm" is easy. By 91%, the goalposts moved to 6.3cm — the same approach behavior that passed the 12.5cm test now fails the 6.3cm test.

"Curriculum squeezing" is just my shorthand for this effect: the curriculum tightens the success criteria faster than the agent improves its precision, so the measured success rate drops even though the agent is actually getting better at the task. The reward going up confirms the agent is getting closer — it just can't consistently hit the tighter threshold.

This is the intended design of the curriculum — it keeps pushing the agent toward higher precision. The problem is that kp=50 with v_cmd_scale=0.5 may not give the controller enough authority to achieve 5cm precision, which is why the next step is enabling residual gains so the agent can learn to temporarily increase stiffness for precise positioning.