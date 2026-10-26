import ast
import operator as op
import re
import time

from botocore.exceptions import BotoCoreError, ClientError, EventStreamError
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from filing_analyst.graph.retriever import query_companies_by_risk, query_facts, search_chunks
from filing_analyst.llm.models import get_chat_llm

_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}
_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        return _ALLOWED_BIN_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        fn = _ALLOWED_FUNCS[node.func.id]
        args = [_eval_ast(a) for a in node.args]
        if node.func.id == "round" and len(args) == 2:
            return float(fn(args[0], int(args[1])))
        return float(fn(*args))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression for financial analysis.

    Supports +, -, *, /, **, %, parentheses, and abs/round/min/max.
    Examples:
      - YoY growth %: '(416161000000 - 391035000000) / 391035000000 * 100'
      - Net margin %: '112010000000 / 416161000000 * 100'
      - Difference:   '416161000000 - 391035000000'
    """
    cleaned = _THOUSANDS_COMMA_RE.sub("", expression).replace("$", "").strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_ast(tree.body)
        return f"{expression} = {result:.6g} (exact: {result})"
    except Exception as exc:
        return f"Calculator error for '{expression}': {exc}"


model = get_chat_llm()

research_agent = create_agent(
    model=model,
    tools=[search_chunks, query_facts, query_companies_by_risk],
    system_prompt=(
        "You are `researcher`, the Neo4j graph research agent. The `supervisor` delegates questions to you.\n\n"
        "INSTRUCTIONS:\n"
        "- GRAPH-FIRST RULE: You MUST call `query_facts`, `search_chunks`, or `query_companies_by_risk` FIRST to check the Neo4j graph before answering.\n"
        "- The graph holds 5 companies (AAPL, MSFT, NVDA, AMZN, TSLA), 3 financial metrics (revenue, net_income, total_assets), 10-K business & risk_factors text chunks, and 10 risk categories.\n"
        "- STRICT SCOPE: Return ONLY the specific information requested. If the user asks only for company names, return ONLY the company names and tickers — do NOT add tables, metrics, fiscal years, or extra facts.\n"
        "- For any fact or text excerpt you include, prefix with `[SOURCE: GRAPH]` and include its `[fact_id]` or `[chunk_id]`.\n"
        "- FALLBACK RULE: If part of the user's question is NOT stored in the Neo4j graph, answer that part briefly and prefix it with `[SOURCE: MODEL_KNOWLEDGE - not present in Neo4j graph]`.\n"
        "- Keep your response brief and direct."
    ),
    name="researcher",
)

analyst_agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt=(
        "You are `analyst`, the financial analyst agent. The `supervisor` delegates analysis and calculations to you after `researcher` gathers evidence.\n\n"
        "INSTRUCTIONS:\n"
        "- Use ONLY the evidence provided in the conversation by `researcher`.\n"
        "- Use the `calculator` tool for all arithmetic (growth rates, margins, differences, rankings, ratios); never calculate in your head.\n"
        "- Cite every graph number you include with its `[fact_id]` and every 10-K text claim with its `[chunk_id]`.\n"
        "- If any part of the answer comes from `[SOURCE: MODEL_KNOWLEDGE - not present in Neo4j graph]`, briefly note that it comes from general knowledge rather than the graph.\n"
        "- Keep your response very short (1-4 sentences) and strictly limited to what the user asked."
    ),
    name="analyst",
)

checkpointer = InMemorySaver()

supervisor = create_supervisor(
    model=model,
    agents=[research_agent, analyst_agent],
    prompt=(
        "You are `supervisor` managing two specialized agents:\n"
        "- `researcher`: Gathers data from the Neo4j 10-K graph (and flags any general knowledge not present in the graph).\n"
        "- `analyst`: Performs calculations using `calculator` and writes the cited analysis.\n\n"
        "WORKFLOW RULES:\n"
        "1. First assign `researcher` to gather evidence from the Neo4j graph (unless the exact answer is already in the current conversation history).\n"
        "2. Only assign `analyst` if the question requires arithmetic calculations or multi-metric comparisons.\n"
        "3. Assign work to one agent at a time; do not call agents in parallel and do not do research or math yourself.\n"
        "4. STRICT BREVITY: Answer ONLY what the user asked in 1-4 short sentences (or a compact bullet list). If they ask for company names, output ONLY the company names. Never add unrequested tables, extra metrics, caveats, or 'I can also help with...' filler. Include `[fact_id]` / `[chunk_id]` citations only for facts/claims actually stated in the final answer."
    ),
    add_handoff_back_messages=False,
    output_mode="full_history",
).compile(checkpointer=checkpointer)

_STATUS_LABELS = {
    "researcher": "  -> Researching in Neo4j graph...",
    "analyst": "  -> Analyzing & running calculations...",
}

_CITATION_TAG_RE = re.compile(
    r"(?:[ ,;]*\[(?:(?:AAPL|MSFT|NVDA|AMZN|TSLA)-[^\]]+|SOURCE:[^\]]+)\])+"
)
_TRAILING_COMMA_WAIT_RE = re.compile(r",\s*$")
_WORD_TOKEN_RE = re.compile(r"\S+\s*|\s+")


def _stream_words(text: str, word_delay: float = 0.018) -> None:
    """Print `text` word-by-word so fast LLM chunks appear as smooth terminal streaming."""
    for token in _WORD_TOKEN_RE.findall(text):
        print(token, end="", flush=True)
        if token.strip():
            time.sleep(word_delay)


def stream_query(user_query: str, thread_id: str = "cli-session-1", verbose: bool = False) -> None:
    """Stream the response to `user_query` with automatic retry on transient Bedrock stream drops.

    - verbose=False (default): shows quiet status lines while sub-agents work and streams ONLY the clean final answer
      (with [chunk_id] / [fact_id] tags stripped).
    - verbose=True: streams the full trace of supervisor, researcher, analyst, tool calls, and [chunk_id]/[fact_id] citations.
    """
    config = {"configurable": {"thread_id": thread_id}}

    for attempt in range(1, 4):
        current_speaker = None
        shown_status: set[str] = set()
        printed_answer_header = False
        bracket_buf = ""
        try:
            for namespace, (msg, _metadata) in supervisor.stream(
                {"messages": [{"role": "user", "content": user_query}]},
                config=config,
                stream_mode="messages",
                subgraphs=True,
            ):
                speaker = namespace[0].split(":")[0] if namespace else "supervisor"

                if verbose:
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    for tc in msg.tool_call_chunks or []:
                        if tc.get("name"):
                            print(f"\n  [tool call by {speaker}] -> {tc['name']}", end="", flush=True)
                    text = msg.text
                    if not text:
                        continue
                    if speaker != current_speaker:
                        current_speaker = speaker
                        print(f"\n\n[{speaker}]: ", end="", flush=True)
                    print(text, end="", flush=True)
                else:
                    if speaker in _STATUS_LABELS:
                        if speaker not in shown_status:
                            shown_status.add(speaker)
                            print(_STATUS_LABELS[speaker], flush=True)
                        continue

                    # In clean mode, stream supervisor text whenever it is speaking to the user (not calling a handoff tool)
                    if speaker == "supervisor" and isinstance(msg, AIMessageChunk):
                        if msg.tool_call_chunks:
                            continue
                        text = msg.text
                        if not text:
                            continue
                        if not printed_answer_header:
                            printed_answer_header = True
                            print("\nAnswer:\n", end="", flush=True)

                        combined = bracket_buf + text
                        bracket_buf = ""
                        # Hold any unclosed '[' or a trailing ', ' right after a citation tag
                        last_open = combined.rfind("[")
                        last_close = combined.rfind("]")
                        if last_open > last_close:
                            bracket_buf = combined[last_open:]
                            combined = combined[:last_open]

                        cleaned_text = _CITATION_TAG_RE.sub("", combined)
                        if last_close != -1 and _TRAILING_COMMA_WAIT_RE.search(cleaned_text):
                            m = _TRAILING_COMMA_WAIT_RE.search(cleaned_text)
                            if m:
                                bracket_buf = cleaned_text[m.start():] + bracket_buf
                                cleaned_text = cleaned_text[:m.start()]
                        if cleaned_text:
                            _stream_words(cleaned_text)

            if not verbose and bracket_buf:
                cleaned_tail = _CITATION_TAG_RE.sub("", bracket_buf)
                if cleaned_tail:
                    _stream_words(cleaned_tail)
            print()
            return
        except (EventStreamError, ClientError, BotoCoreError) as exc:
            if attempt < 3:
                wait_s = 2 * attempt
                print(f"\n  [Bedrock transient stream error ({type(exc).__name__}); retrying in {wait_s}s...]", flush=True)
                time.sleep(wait_s)
            else:
                print(f"\n  [Bedrock error after 3 attempts: {exc}]", flush=True)


if __name__ == "__main__":
    verbose_mode = False
    print("SEC Filings Analyst (Claude Sonnet 5.5)")
    print("  Commands: type '/verbose' to toggle full agent trace (currently: OFF - clean answer mode)")
    print("            type 'q' or 'exit' to quit")
    while True:
        try:
            query = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        if query.lower() == "/verbose":
            verbose_mode = not verbose_mode
            state_label = "ON (full multi-agent trace & source tags)" if verbose_mode else "OFF (clean final answer only)"
            print(f"Verbose trace mode: {state_label}")
            continue
        stream_query(query, verbose=verbose_mode)
