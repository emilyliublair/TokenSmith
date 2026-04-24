"""
Query enhancement techniques for improved retrieval (use only one):
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer for better retrieval
- Query Enrichment: LLM-based query expansion
"""

import textwrap
from typing import Optional
from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning


def generate_hypothetical_document(
    query: str,
    model_path: str,
    max_tokens: int = 100,
    **llm_kwargs
) -> str:
    """
    HyDE: Generate a hypothetical answer to improve retrieval quality.
    Concept: Hypothetical answers are semantically closer to actual documents than queries.
    Ref: https://arxiv.org/abs/2212.10496
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a database systems expert. Generate a concise, technical answer using precise database terminology.
        Write in the formal academic style of Database System Concepts (Silberschatz, Korth, Sudarshan).
        Use specific terms for: relational model concepts (relations, tuples, attributes, keys, schemas), 
        SQL and query languages, transactions (ACID properties, concurrency control, recovery), 
        storage structures (indexes, B+ trees), normalization (functional dependencies, normal forms), 
        and database design (E-R model, decomposition).
        Focus on definitions, mechanisms, and technical accuracy rather than examples.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Generate a precise and a concise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    hypothetical = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        **llm_kwargs
    )
    
    return hypothetical.strip()

def correct_query_grammar(
    query: str,
    model_path: str,
    **llm_kwargs
) -> str:
    """
    Corrects spelling and grammatical errors in the query to improve keyword matching.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that corrects search queries.
        Your task is to correct any spelling or grammatical errors in the user's query.
        Do not answer the question. Output ONLY the corrected query.
        <|im_end|>
        <|im_start|>user
        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    corrected_query = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=len(query.split()) * 2,
        temperature=0,
        **llm_kwargs
    )

    # If model returns empty or hallucinated long text, return original
    cleaned = corrected_query["choices"][0]["text"].strip()
    if not cleaned or len(cleaned) > len(query) * 2:
        return query

    return cleaned

def expand_query_with_keywords(
    query: str,
    model_path: str,
    max_tokens: int = 64,
    **llm_kwargs
) -> str:
    """
    Query Expansion: Generates related keywords and synonyms.
    This helps retrieval when the user uses different vocabulary than the documents.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search optimization expert.
        Generate 3 alternative versions of the user's query using synonyms and related technical terms.
        Output the alternative queries separated by newlines. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    expansion = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.5,
        **llm_kwargs
    )

    # Combine original query with expansion
    query_lines = [query]
    query_lines.extend([line.strip() for line in expansion["choices"][0]["text"].split('\n') if line.strip()])

    # Remove numbering if present
    query_lines = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in query_lines]

    return query_lines


def decompose_complex_query(
    query: str,
    model_path: str,
    **llm_kwargs
) -> list[str]:
    """
    Breaks a complex multi-part question into sub-questions.
    Useful for tasks where a single retrieval might miss some parts of the answer.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        Break the following complex question into simple, single-step sub-questions.
        If the question is already simple, just output the original question.
        Output each sub-question on a new line. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Complex Question: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=128,
        temperature=0.0,
        **llm_kwargs
    )

    sub_questions = [line.strip() for line in output["choices"][0]["text"].split('\n') if line.strip()]

    # Remove numbering if present
    sub_questions = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in sub_questions]

    return sub_questions

def contextualize_query(
    query: str,
    history: list[dict],
    model_path: str,
    max_tokens: int = 128,
    **llm_kwargs
) -> str:
    """
    Rewrites a query to be standalone based on chat history.
    """
    if not history:
        return query

    # Format history into a compact string
    # We expect history to be list of dicts: [{"role": "user", "content": "..."}, ...]
    conversation_text = ""
    for turn in history[-4:]: # Only look at last 2 turns
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        conversation_text += f"{role}: {content}\n"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant. Your task is to rewrite the user's "Follow Up Input" to be a standalone question by replacing pronouns (it, they, this, that) with specific nouns from the "Chat History".
        
        Examples:
        History: 
        User: What is BCNF?
        Assistant: It is a normal form used in database normalization.
        Input: Why is it useful?
        Output: Why is BCNF useful?
        
        History:
        User: Explain the ACID properties.
        Assistant: ACID stands for Atomicity, Consistency, Isolation, Durability.
        Input: Give me an example of the first one.
        Output: Give me an example of Atomicity.

        History:
        User: Who created Python?
        Assistant: Guido van Rossum.
        Input: what is sql?
        Output: what is sql?
        <|im_end|>
        <|im_start|>user
        Chat History:
        {conversation_text}
        
        Follow Up Input: {query}
        
        Output:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs
    )

    rewritten = output["choices"][0]["text"].strip()
    
    # If model hallucinates or errors, fall back to original query
    if not rewritten or len(rewritten) > len(query) * 2:
        return query
        
    return rewritten

def generate_doc2query_questions(
    chunk_text: str,
    model_path: str,
    num_questions: int = 3,
    max_tokens: int = 150,
    **llm_kwargs
) -> list[str]:
    """
    Doc2Query: Generates synthetic questions for a given text chunk.
    These questions are appended to the document during index building 
    to improve retrieval recall for complex or poorly phrased user queries.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are generating realistic search queries that students type when they are confused about a database concept.

        Rules:
        - Write exactly 5 questions, one per line
        - First 3 questions: student has NO technical vocabulary. Use everyday words only. Focus on what they observe or want to do.
        - Last 2 questions: student knows 1-2 terms but is confused about details.
        - Never copy phrases from the text.
        - Never output meta-commentary, labels, or formatting. Only output the 5 questions.

        Good example output for a chunk about the LIKE operator:
        How do I search for words that start with a certain letter in my database?
        My query is returning everything instead of just matching names, what did I do wrong?
        How do I find rows where a column contains a specific word somewhere in it?
        How does pattern matching work in SQL with wildcards?
        What is the difference between LIKE and equals when filtering string columns?

        Bad example output (never do this):
        Describe the GOAL or SYMPTOM in plain language
        No technical terms from the text
        What is a SYMPTOM in plain language?
        <|im_end|>
        <|im_start|>user
        Text:
        {chunk_text}
        <|im_end|>
        <|im_start|>assistant
        """)

    # Increase temperature to 0.7 or 0.8 for more diverse phrasing
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=500, 
        temperature=0.8, 
        **llm_kwargs
    )

    # Extract the generated text safely based on standard llama_cpp dictionary format
    if isinstance(output, dict) and "choices" in output:
        result_text = output["choices"][0]["text"].strip()
    else:
        result_text = str(output).strip()
    
    # Clean the output and strip any bullet points or numbers the LLM might have added
    raw_questions = result_text.split('\n')
    cleaned_questions = [
        q.lstrip("-*1234567890. ").strip() 
        for q in raw_questions 
        if q.strip()
    ]
    
    return cleaned_questions[:num_questions]