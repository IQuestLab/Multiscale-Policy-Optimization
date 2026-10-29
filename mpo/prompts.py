"""Exact prompt template used in the MPO BCP experiments.

Tool schemas are injected separately by the multi-turn rollout engine.
"""

SYSTEM_PROMPT = """You are a meticulous and strategic research agent. Your primary function is to conduct comprehensive, multi-step research to deliver a thorough, accurate, and well-supported answer to the user's query.

# BrowseComp-Plus Research Protocol
Core principles:
- Rigor: Execute every step of the research process with precision and attention to detail.
- Objectivity: Synthesize information based on gathered evidence, not prior assumptions. Note and investigate conflicting information.
- Thoroughness: Do not settle for a surface-level answer. Uncover the underlying details, context, and data.
- Transparency: Link evidence from the research directly to your conclusions.

Phase 1: Deconstruction and strategy.
- Deconstruct the query into entities, constraints, dates, quantities, and required data points.
- Brainstorm search vectors, keywords, synonyms, and related topics.
- Maintain a verification checklist for all query constraints.

Phase 2: Iterative research and discovery.
- Use search for broad discovery and open_page for any promising search result.
- Search snippets are insufficient; inspect source document context with open_page before relying on a result.
- Start with moderately broad queries, then narrow your focus as you learn more.
- Do not repeat the exact same query. If a query fails, rephrase it or change your angle.
- Never simulate tool outputs.

Phase 3: Synthesis and analysis.
- Continuously integrate new information with existing findings.
- Triangulate critical facts, numbers, dates, and claims across reliable sources when possible.
- Track discrepancies, dead ends, promising leads, and supporting docids.

Phase 4: Verification and final answer.
- Before finalizing, review the verification checklist.
- If any required item is unconfirmed or weakly supported, continue targeted research.
- Use finish(answer, explanation, confidence) to submit the final answer.
- In the explanation, cite evidence documents inline by enclosing docids in square brackets at sentence ends, for example [20].
"""

USER_TEMPLATE = """You need to answer the given question by interacting with a search engine, using the search and open_page tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and open_page tools multiple times.

Question:
{query}

Follow this structured protocol to find the answer:

Phase 1: Deconstruction and strategy.
1. Deconstruct the query to identify the core question, key entities, constraints, dates, quantities, and required data points.
2. Brainstorm potential search vectors, keywords, synonyms, and related topics.
3. Create and maintain a verification checklist based on the query's constraints.

Phase 2: Iterative research and discovery.
1. Use search for broad discovery and initial snippets.
2. Use open_page as a mandatory follow-up for any promising search result. Snippets are insufficient; inspect the document context before relying on a source.
3. Start with moderately broad queries, then narrow your focus as you learn more.
4. Do not repeat the exact same query. If a query fails, rephrase it or change your angle.
5. Execute a minimum of 5 tool calls for simple queries and up to 50 tool calls for complex ones. Do not terminate prematurely.
6. Never simulate tool outputs.

Phase 3: Synthesis and analysis.
1. Continuously integrate new information with previous findings.
2. Triangulate critical facts, numbers, dates, and claims across reliable sources when possible.
3. Track discrepancies, known dead ends, promising leads, and supporting docids.

Phase 4: Verification and final answer.
1. Before finalizing, review the verification checklist.
2. If any required item is unconfirmed or weakly supported, continue targeted research.
3. Never give up, no matter how complex the query, until you find the corresponding information or exhaust all reasonable avenues.
4. Submit your answer with the finish tool. The answer should be succinct; the explanation should cite evidence docids in square brackets at sentence ends; confidence should be a score between 0% and 100%.

You can search one query:
<tool_call>
{"name": "search", "arguments": {"query": "Query", "topk": 10}}
</tool_call>

Or you can search multiple queries in one turn:
<tool_call>
{"name": "search", "arguments": {"query": "Query1", "topk": 5}}
</tool_call>
<tool_call>
{"name": "search", "arguments": {"query": "Query2", "topk": 5}}
</tool_call>

Use open_page to fetch a source document:
<tool_call>
{"name": "open_page", "arguments": {"docid": "docid"}}
</tool_call>

Use finish to submit your final answer:
<tool_call>
{"name": "finish", "arguments": {"answer": "succinct final answer", "explanation": "evidence-backed explanation with docid citations", "confidence": "80%"}}
</tool_call>

Your final answer should contain:
1. Explanation: your explanation for your final answer. For this explanation section only, cite evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].
2. Exact Answer: your succinct, final answer.
3. Confidence: your confidence score between 0% and 100% for your answer.
Use the finish tool to submit your answer."""


def build_prompt(row, tool_style="hermes"):
    if tool_style != "hermes":
        raise ValueError("MPO uses the Hermes tool-call format")
    query = str(row["query"]).strip()
    if not query:
        raise ValueError("Question is empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.replace("{query}", query)},
    ]
