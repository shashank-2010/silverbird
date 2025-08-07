

class SessionContextManager:

    _session_store = {}

    @classmethod
    def save_context(cls, session_id: str, context: dict):
        cls._session_store[session_id] = context

    @classmethod
    def get_context(cls, session_id: str) -> dict:
        return cls._session_store.get(session_id)

    @classmethod
    def clear_context(cls, session_id: str):
        if session_id in cls._session_store:
            del cls._session_store[session_id]

    @classmethod
    def has_context(cls, session_id: str) -> bool:
        return session_id in cls._session_store
