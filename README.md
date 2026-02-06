You can read more about the approach here: https://sarooshkhan24.substack.com/p/solving-arc-with-test-time-adaptation

## Claude Opus 4.6 Results

Ran my January 27th attempt on Claude Opus 4.6 by @AnthropicAI.

Achieved 86.2% at $10.34/task. (reasoning set to enabled in OpenRouter)

Some key differences I noticed vs GPT 5.2(xhigh):

- 4.6 uses way more tokens in reasoning + is more likely to reason in all tool calls.
- Regressed on 4 tasks that GPT 5.2 solved but outperformed overall.
- Costs around 5x more vs GPT 5.2 but uses lesser number of tool calls to converge
- Very low reliance on supplementary solvers. (Running only the iterative solver gets you 84%)
