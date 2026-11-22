"""Chat history: conversations owned by one resident (tenant + user).

Lives beside `rag/`, not inside it: retrieval stays a plain function, and the
phase 2 agent will read and write the same conversations.
"""
