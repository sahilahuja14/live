from .customers import register_customer_routes
from .scoring import register_scoring_routes
from .system import register_system_routes

__all__ = [
    "register_customer_routes",
    "register_scoring_routes",
    "register_system_routes",
]
