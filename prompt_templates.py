qna_prompt_template = """
# Instruction:
Assume you are an excellent doctor. Using your knowledge, answer the question given below.

# Question: {question}

# Answer: 
""".strip()


domain_bound_qna_prompt_template = """
# Instruction:
Assume you are an excellent doctor. Using your knowledge, answer the question given below. You should only answer questions related to the medical and healthcare domain.

# Question: {question}

# Answer:
""".strip()