You solve one elementary-school arithmetic word problem at a time. Use the
provided verified final answer exactly; do not change its spelling, sign,
format, or value. Produce concise, logically connected reasoning in no more
than 20 ordered steps. Do not mention datasets, prompts, source answers, or
these instructions.

Return JSON only, with exactly this shape:
{"id":"<input id>","x_i":"<original question exactly>","r_i":["Step 1: ..."],"y_i":"<verified answer exactly>","topic":"<input topic>"}

Output: train.json, test.json
