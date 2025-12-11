import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

cliente = genai.Client(api_key=api_key)


def enhanced_spell_query(query):
    response = cliente.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.

        Query: "{query}"

        If no errors, return the original query.
        Corrected:""",
    )
    return response.text.strip()


def enhanced_rewrite_query(query):
    response = cliente.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:""",
    )
    return response.text.strip()


def enhanced_expand_query(query):
    response = cliente.models.generate_content(
        model="gemma-3-12b-it",
        contents=f"""Expand this movie search query with related terms.
        
        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        This will be appended to the original query.
        
        Examples:
        
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"
        
        Query: "{query}"
        """,
    )
    return response.text.strip()


def individual_rerank(query, title, description):
    response = cliente.models.generate_content(
        model="gemma-3-12b-it",
        contents=f"""Rate how well this movie matches the search query.
        Query: "{query}"
        Movie: {title} - {description}
        
        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness
        
        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.
        
        Score:""",
    )
    return response.text.strip()


def batch_rerank(query, movies):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {movies}

        Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

        [75, 12, 34, 2, 1]
        """,
    )
    return response.text.strip()


def evaluate_results(query, results):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {results}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers out than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]""",
    )
    return response.text.strip()


def generate_answer(query, movies):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Documents:
        {movies}

        Provide a comprehensive answer that addresses the query:""",
    )
    return response.text.strip()


def summarize_movies(query, movies):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:
        {movies}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """,
    )
    return response.text.strip()


def summarize_citations(query, movies):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Answer the question or provide information based on the provided documents.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {movies}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:""",
    )
    return response.text.strip()


def question_summarize(question, movies):
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Answer the user's question based on the provided movies that are available on Hoopla.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Question: {question}

        Documents:
        {movies}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:""",
    )
    return response.text.strip()


def generate_multimodal(image, mime, query):
    parts = [
        """
        Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
        - Synthesize visual and textual information
        - Focus on movie-specific details (actors, scenes, style, etc.)
        - Return only the rewritten query, without any additional commentary
        """,
        types.Part.from_bytes(data=image, mime_type=mime),
        query.strip(),
    ]
    response = cliente.models.generate_content(
        model="gemini-2.5-flash-lite", contents=parts
    )
    text = response.text.strip()
    tokens = response.usage_metadata.total_token_count
    return text, tokens
