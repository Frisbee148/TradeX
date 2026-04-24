"""
FastAPI application for the MEVerse Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import MeverseAction, MeverseObservation
    from .meverse_environment import MeverseEnvironment
except ModuleNotFoundError:
    from models import MeverseAction, MeverseObservation
    from server.meverse_environment import MeverseEnvironment


app = create_app(
    MeverseEnvironment,
    MeverseAction,
    MeverseObservation,
    env_name="meverse",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
