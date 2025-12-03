import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
# print(f"using key: {api_key[:6]}...")

cliente = genai.Client(api_key=api_key)
# response = cliente.models.generate_content(
#     model="gemini-2.0-flash-001",
#     contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
# )

# print(response.text)

# print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
# print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")


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
        model="gemini-2.0-flash-001",
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
