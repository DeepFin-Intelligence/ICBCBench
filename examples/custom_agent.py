"""A runnable, offline agent adapter demonstrating ICBCBench's call contract.

Replace ``generate`` with your own research workflow before measuring a model.
This example deliberately returns synthetic placeholders, not task solutions.
The subsequent scoring pipeline still requires judge API access.
"""


def generate(task):
    """Accept one task dict and return a completed response dict.

    ``task`` contains id, track, language, model, and formatted messages. It does
    not contain the reference answer or expert rubric. Your implementation can
    pass task["messages"] to its model and run its own search/retrieval tools.
    Raise an exception on a failed request instead of returning an empty result.
    """
    if task["track"] == "objective":
        if task["language"] == "zh":
            content = (
                "解析：这是用于验证接口的合成示例，没有实际解答本题。\n"
                "确切答案：未作答\n置信度：0%"
            )
        else:
            content = (
                "Explanation: Synthetic integration example; the task was not solved.\n"
                "Exact Answer: Not answered\nConfidence: 0%"
            )
    elif task["track"] == "subjective":
        if task["language"] == "zh":
            content = (
                "# 合成示例报告\n\n"
                "这是用于验证接口的占位报告，没有开展研究或查阅资料，"
                "不代表任何模型的评测结果。真实 Agent 应在报告中使用可访问的"
                "来源 URL，将事实主张与引用对应起来。"
            )
        else:
            content = (
                "# Synthetic example report\n\n"
                "This placeholder demonstrates the adapter contract. No research "
                "or source retrieval was performed, and this is not a model result. "
                "A real agent should link factual claims to accessible source URLs "
                "in the report."
            )
    else:
        raise ValueError(f"Unknown track: {task['track']}")
    return {"content": content, "citations": []}
