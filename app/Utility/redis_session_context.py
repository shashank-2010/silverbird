import json
import redis
import os
from datetime import datetime

class RedisSessionContextManager:
    def __init__(self):
        self.redis_client = redis.StrictRedis(
            host="redis-12070.c301.ap-south-1-1.ec2.redns.redis-cloud.com",
            port=12070,
            username="default",
            password="Xc5Y1omrH9XZAsGkX5tgpQGGMCQIlesb",
            decode_responses=True  
        )

    def save_context(self, user_id: str, context: dict):
        """Save chat context for a user."""
        self.redis_client.set(user_id, json.dumps(context))
        # Optional: expire after 24 hours
        self.redis_client.setex(user_id, 86400, json.dumps(context))

    def get_context(self, user_id: str) -> dict:
        """Retrieve chat context for a user."""
        data = self.redis_client.get(user_id)
        return json.loads(data) if data else None

    def clear_context(self, user_id: str):
        """Clear chat context for a user."""
        self.redis_client.delete(user_id)

    def has_context(self, user_id: str) -> bool:
        """Check if a user has a saved context."""
        return self.redis_client.exists(user_id) > 0
    
    def save_message(self, user_id, role, message):
        self.redis_client.rpush(f"chat_history:{user_id}", json.dumps({
            "role": role,   # "user" or "assistant"
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def get_chat_history(self, user_id):
        messages = self.redis_client.lrange(f"chat_history:{user_id}", 0, -1)
        return [json.loads(m) for m in messages]

    def clear_chat_history(self, user_id: str):
        self.redis_client.delete(f"chat_history:{user_id}")

    def get_all_messages(self, session_id):
        raw_messages = self.redis_client.lrange(f"chat:{session_id}", 0, -1)
        messages = [json.loads(msg) for msg in raw_messages]
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
