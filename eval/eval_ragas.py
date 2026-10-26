import json
from datasets import Dataset
from langchain_core.messages import AIMessage, ToolMessage
from ragas import evaluate
from ragas.metrics.collections import answer_relevancy, context_precision, context_recall, faithfulness

from filing_analyst.agent.agents import _CITATION_TAG_RE, supervisor
from filing_analyst.config import PROJECT_ROOT
from filing_analyst.llm.models import get_chat_llm, get_embeddings

EVAL_DIR = PROJECT_ROOT / "eval"
items = json.loads((EVAL_DIR / "eval_dataset.json").read_text(encoding="utf-8"))

questions, answers, contexts, ground_truths = [], [], [], []

for item in items:
    print(f"Running {item['id']}: {item['question']}")
    res = supervisor.invoke(
        {"messages": [{"role": "user", "content": item["question"]}]},
        config={"configurable": {"thread_id": item["id"]}},
    )
    msgs = res["messages"]

    # Retrieved contexts from researcher/analyst tool calls (exclude handoff tool messages)
    ctxs = [
        str(m.content)
        for m in msgs
        if isinstance(m, ToolMessage)
        and m.content
        and not str(getattr(m, "name", "")).startswith("transfer_to_")
    ]
    # Final answer from supervisor
    ans = next(
        (_CITATION_TAG_RE.sub("", m.text).strip() for m in reversed(msgs) if isinstance(m, AIMessage) and m.text),
        "",
    )

    questions.append(item["question"])
    answers.append(ans)
    contexts.append(ctxs or ["No matching data in graph."])
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=get_chat_llm(),
    embeddings=get_embeddings(),
)

print(result)
result.to_pandas().to_json(EVAL_DIR / "eval_results.json", orient="records", indent=2)
