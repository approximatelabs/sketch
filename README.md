# sketch

# Currently a work in progress.

Co-pilot for pandas users, AI that understands the content of data, greatly enhancing the relevance of suggestions. 

Adding data context to AI code-writing assistants, usable in any Jupyter in seconds.

```
pip install sketch
```

## Example (gif) 
```python
import sketch
...
df.sketch.howto("Check for any duplicate rows, and keep the first one based on the time feature")
```

## It would also be pretty good to compare this to copilot directly
(Show a copilot suggestion with comment block, and its output)
(Show a GPT-3 codex response)

## How to run with your own OpenAI API key

If you add `OPENAI_API_KEY` environment variable and `LOCAL_LAMBDA_PROMPT=True`, then sketch will run the prompts locally, directly using your API key with openAI's endpoints. 

## How it works

Sketch uses efficient approximation algorithms (data sketches) to quickly summarize your data, and feed that information into language models. Right now it does this by summarizing the columns and writing these summary statistics as additional context to be used by the code-writing prompt. In the future we hope to feed these sketches directly into custom made "data + language" foundation models. 

