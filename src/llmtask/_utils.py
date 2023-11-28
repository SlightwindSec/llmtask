

choices = ["A", "B", "C", "D"]

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = len(choices)
    for i in range(k):
        prompt += "\n{}. {}".format(choices[i], df.iloc[idx, i + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(dev_df, subject, n):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        " " + " ".join(subject.split("_"))
    )
    n = min(n, dev_df.shape[0]) # n-shot prompts
    for i in range(n):
        prompt += format_example(dev_df, i)
    return prompt
