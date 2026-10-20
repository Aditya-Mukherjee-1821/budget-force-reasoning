You are annotating elementary-school arithmetic word problems.

For each supplied record, classify the question into exactly one of these topic
strings:
addition_subtraction_counting, multiplication_division, fractions,
ratios_and_proportions, percentages_discount_profit, rates_time_distance_work,
money_and_unit_conversion, averages_and_mixtures, geometry_measurement.

Return JSON only, with exactly this shape:
{"id":"<input id>","topic":"<one allowed topic>","topic_confidence":0.0}

`topic_confidence` must be a number from 0 to 1. Do not solve, rewrite,
correct, quote, or otherwise modify the question or its answer.

Output: gpt_response_train.json, gpt_response_test.json
