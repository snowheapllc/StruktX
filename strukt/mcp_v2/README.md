# FastMCP v2 Integration for StruktX

This module provides FastMCP v2 integration for StruktX, exposing handlers with `mcp_*` prefixed methods as MCP tools. It maintains Strukt's framework-agnostic philosophy while leveraging FastMCP's modern features.

## Key Features

- **Convention-Based Tool Discovery**: Automatically exposes handlers with `mcp_*` prefixed methods as MCP tools
- **Type-Safe Schema Generation**: Leverages Python type hints for automatic input/output schema generation
- **Middleware Bridging**: Seamlessly integrates Strukt's middleware system with FastMCP
- **Multiple Transport Support**: Supports stdio, SSE, and HTTP transports
- **Authentication Integration**: Built-in API key authentication support
- **Framework Agnostic**: Maintains Strukt's swappable, framework-agnostic design

## Quick Start

### 1. Install Dependencies

```bash
pip install fastmcp>=2.12.5
```

### 2. Create a Handler with MCP Tools

```python
from strukt.interfaces import Handler
from strukt.types import HandlerResult, InvocationState
from typing import Dict, Any

class MyHandler(Handler):
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        # Your regular handler logic
        return HandlerResult(response="Hello from handler", status="success")
    
    def mcp_get_time(self, timezone: str = "UTC") -> Dict[str, Any]:
        """Get current time in specified timezone."""
        import datetime
        import pytz
        
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        
        return {
            "time": now.isoformat(),
            "timezone": timezone,
            "timestamp": now.timestamp()
        }
    
    def mcp_calculate(self, operation: str, a: float, b: float) -> float:
        """Perform basic mathematical operations."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float('inf')
        }
        
        if operation not in operations:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return operations[operation](a, b)
```

### 3. Create and Run the FastMCP Server

```python
from strukt.mcp_v2 import StruktFastMCPServer, run_stdio

# Create your handlers
handlers = {
    "my_handler": MyHandler()
}

# Create server
server = StruktFastMCPServer(
    handlers=handlers,
    server_name="my-struktx-mcp",
    include_handlers=["my_handler"]  # Optional: filter handlers
)

# Run over stdio (for MCP clients)
run_stdio(server)
```

### 4. Run with Different Transports

```python
from strukt.mcp_v2 import run_sse, run_http

# Run over SSE (Server-Sent Events)
run_sse(server, host="localhost", port=8000)

# Run over HTTP
run_http(server, host="localhost", port=8000)
```

## Configuration

### Basic Configuration

```python
from strukt.config import StruktConfig, MCPv2Config

config = StruktConfig()
config.mcp_v2.enabled = True
config.mcp_v2.server_name = "my-struktx-mcp"
config.mcp_v2.transport = "stdio"  # or "sse", "http"
config.mcp_v2.include_handlers = ["my_handler"]
```

### Authentication Configuration

```python
# Enable API key authentication
config.mcp_v2.auth_api_key.enabled = True
config.mcp_v2.auth_api_key.header_name = "x-api-key"
config.mcp_v2.auth_api_key.env_var = "STRUKTX_MCP_API_KEY"

# Set the API key in environment
import os
os.environ["STRUKTX_MCP_API_KEY"] = "your-secret-api-key"
```

### YAML Configuration

```yaml
mcp_v2:
  enabled: true
  server_name: "my-struktx-mcp"
  transport: "stdio"
  include_handlers:
    - "my_handler"
    - "time_handler"
  auth_api_key:
    enabled: true
    header_name: "x-api-key"
    env_var: "STRUKTX_MCP_API_KEY"
  sse_host: "localhost"
  sse_port: 8000
  http_host: "localhost"
  http_port: 8000
```

## Tool Discovery

### Convention-Based Discovery

FastMCP v2 automatically discovers tools from handlers using the `mcp_*` prefix:

```python
class MyHandler(Handler):
    # This becomes a tool named "my_handler_get_time"
    def mcp_get_time(self, timezone: str = "UTC") -> Dict[str, Any]:
        """Get current time in specified timezone."""
        # Implementation...
    
    # This becomes a tool named "my_handler_calculate"
    def mcp_calculate(self, operation: str, a: float, b: float) -> float:
        """Perform basic mathematical operations."""
        # Implementation...
    
    # This becomes a tool named "my_handler_process_data"
    def mcp_process_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a list of data items."""
        # Implementation...
```

### Tool Naming

Tools are automatically named using the pattern: `{handler_key}_{method_suffix}`

- Handler key: `my_handler`
- Method: `mcp_get_time`
- Tool name: `my_handler_get_time`

### Type Hints and Schema Generation

The system automatically generates JSON schemas from Python type hints:

```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class UserData(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

def mcp_process_users(self, users: List[UserData], filter_age: int = 18) -> Dict[str, Any]:
    """Process a list of users with age filtering."""
    # Implementation...
```

This generates:
- Input schema with `users` (array of UserData objects) and `filter_age` (integer, default 18)
- Output schema for the returned dictionary

## Complete Examples

### Example 1: Basic Handler with MCP Tools

```python
from strukt.interfaces import Handler
from strukt.types import HandlerResult, InvocationState
from typing import Dict, Any

class ExampleHandler(Handler):
    """Example handler showing how to create MCP tools with mcp_* methods."""
    
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        """Regular handler method - not exposed as MCP tool."""
        return HandlerResult(response="Hello from handler", status="success")
    
    def mcp_get_time(self, timezone: str = "UTC") -> Dict[str, Any]:
        """Get current time - automatically becomes MCP tool 'example_get_time'."""
        import datetime
        import pytz
        
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        
        return {
            "time": now.isoformat(),
            "timezone": timezone,
            "timestamp": now.timestamp()
        }
    
    def mcp_calculate(self, operation: str, a: float, b: float) -> float:
        """Calculate - automatically becomes MCP tool 'example_calculate'."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float('inf')
        }
        
        if operation not in operations:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return operations[operation](a, b)
```

### Example 2: Handler with Pydantic Models

```python
from pydantic import BaseModel
from typing import List, Dict, Any

class UserData(BaseModel):
    name: str
    age: int
    email: str | None = None

class UserHandler(Handler):
    """Handler demonstrating Pydantic model usage in MCP tools."""
    
    def __init__(self):
        self.users: List[UserData] = []
    
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response="User handler", status="success")
    
    def mcp_add_user(self, user: UserData) -> Dict[str, Any]:
        """Add user - automatically becomes MCP tool 'user_add_user'."""
        self.users.append(user)
        return {
            "success": True,
            "user": user.model_dump(),
            "total_users": len(self.users)
        }
    
    def mcp_list_users(self, min_age: int = 0) -> Dict[str, Any]:
        """List users - automatically becomes MCP tool 'user_list_users'."""
        filtered_users = [u for u in self.users if u.age >= min_age]
        return {
            "users": [u.model_dump() for u in filtered_users],
            "count": len(filtered_users),
            "min_age": min_age
        }
```

### Example 3: Complete Application Setup

```python
from strukt.mcp_v2 import StruktFastMCPServer, run_stdio
from strukt.config import StruktConfig
import os

# 1. Create configuration
config = StruktConfig()

# Configure MCP v2
config.mcp_v2.enabled = True
config.mcp_v2.server_name = "my-app-mcp"
config.mcp_v2.transport = "stdio"  # Use stdio for MCP clients
config.mcp_v2.include_handlers = ["example", "user"]
config.mcp_v2.auth_api_key.enabled = True
config.mcp_v2.auth_api_key.env_var = "MY_APP_MCP_API_KEY"

# Set API key
os.environ["MY_APP_MCP_API_KEY"] = "my-secret-key"

# 2. Create handlers
handlers = {
    "example": ExampleHandler(),
    "user": UserHandler()
}

# 3. Create server
server = StruktFastMCPServer(
    handlers=handlers,
    config=config.mcp_v2
)

# 4. Run server
print(f"Starting MCP server with {len(server.registered_tools)} tools:")
for tool_name in server.registered_tools.keys():
    print(f"  - {tool_name}")

# In production, you would run:
# run_stdio(server)
```

## Migration from MCP v1

### Key Differences

| Feature | MCP v1 (OLD) | MCP v2 (NEW) |
|---------|--------------|--------------|
| **Configuration** | `mcp:` | `mcp_v2:` |
| **Tool Discovery** | Explicit configuration | Automatic via `mcp_*` methods |
| **Method Naming** | Any method name | Must start with `mcp_` |
| **Schema Generation** | Manual configuration | Automatic from type hints |
| **Transport Options** | Limited (mainly ASGI) | stdio, SSE, HTTP |
| **Server Class** | `MCPServerApp` | `StruktFastMCPServer` |
| **Runtime** | External (fast-agent) | Built-in (FastMCP) |

### Migration Steps

1. **Update Dependencies**:
   ```bash
   pip install fastmcp>=2.12.5
   ```

2. **Update Configuration**:
   ```yaml
   # OLD (v1)
   mcp:
     enabled: true
     tools:
       my_handler:
         - name: "get_time"
           method_name: "get_time"
   
   # NEW (v2)
   mcp_v2:
     enabled: true
     include_handlers:
       - "my_handler"
   ```

3. **Update Handler Methods**:
   ```python
   # OLD (v1)
   def get_time(self, timezone: str) -> str:
       # Implementation...
   
   # NEW (v2)
   def mcp_get_time(self, timezone: str) -> Dict[str, Any]:
       # Implementation...
   ```

4. **Update Imports**:
   ```python
   # OLD (v1)
   from strukt.mcp import MCPServerApp, build_fastapi_app
   
   # NEW (v2)
   from strukt.mcp_v2 import StruktFastMCPServer, build_fastapi_app, run_stdio
   ```

5. **Update Server Creation**:
   ```python
   # OLD (v1)
   server = MCPServerApp(
       server_name="my-server",
       handlers=handlers,
       include_handlers=["my_handler"],
       memory=memory,
       api_key_auth=auth_config
   )
   
   # NEW (v2)
   server = StruktFastMCPServer(
       handlers=handlers,
       server_name="my-server",
       include_handlers=["my_handler"],
       config=config.mcp_v2
   )
   ```

### Before/After Comparison

#### BEFORE: MCP v1 Implementation (DEPRECATED)

```yaml
# config.yaml (OLD)
mcp:
  enabled: true
  server_name: "my-mcp-server"
  include_handlers:
    - "time_handler"
    - "device_handler"
  auth_api_key:
    enabled: true
    header_name: "x-api-key"
    env_var: "STRUKTX_MCP_API_KEY"
  tools:
    time_handler:
      - name: "get_time"
        method_name: "get_time"
        description: "Get current time"
        parameters_schema:
          type: "object"
          properties:
            timezone:
              type: "string"
              default: "UTC"
          required: []
```

```python
# Handler with explicit MCP methods (OLD)
class TimeHandler(Handler):
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response="Time handler", status="success")
    
    def get_time(self, timezone: str = "UTC") -> str:
        # This method was explicitly configured in mcp.tools
        import datetime
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return now.isoformat()
```

```python
# OLD server creation
from strukt.mcp import MCPServerApp, build_fastapi_app

server = MCPServerApp(
    server_name="my-mcp-server",
    handlers=handlers,
    include_handlers=["time_handler", "device_handler"],
    memory=memory,
    api_key_auth=APIKeyAuthConfig(
        header_name="x-api-key",
        env_var="STRUKTX_MCP_API_KEY"
    )
)
```

#### AFTER: FastMCP v2 Implementation (RECOMMENDED)

```yaml
# config.yaml (NEW)
mcp_v2:
  enabled: true
  server_name: "my-fastmcp-server"
  transport: "stdio"  # stdio, sse, or http
  include_handlers:
    - "time_handler"
    - "device_handler"
  auth_api_key:
    enabled: true
    header_name: "x-api-key"
    env_var: "STRUKTX_MCP_API_KEY"
  sse_host: "localhost"
  sse_port: 8000
  http_host: "localhost"
  http_port: 8000
  # Note: No explicit tool configuration needed!
  # Tools are auto-discovered from mcp_* methods
```

```python
# Handler with mcp_* methods (NEW)
class TimeHandler(Handler):
    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        return HandlerResult(response="Time handler", status="success")
    
    def mcp_get_time(self, timezone: str = "UTC") -> Dict[str, Any]:
        # This automatically becomes MCP tool 'time_get_time'
        import datetime
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        return {
            "time": now.isoformat(),
            "timezone": timezone,
            "timestamp": now.timestamp()
        }
```

```python
# NEW server creation
from strukt.mcp_v2 import StruktFastMCPServer, run_stdio, run_sse, run_http

server = StruktFastMCPServer(
    handlers=handlers,
    server_name="my-fastmcp-server",
    include_handlers=["time_handler", "device_handler"],
    config=config.mcp_v2
)

# Multiple transport options
run_stdio(server)  # For MCP clients
run_sse(server, host="localhost", port=8000)  # For web clients
run_http(server, host="localhost", port=8000)  # For HTTP clients
```

## Middleware Integration

### Strukt Middleware Bridging

Your existing Strukt middleware automatically works with FastMCP:

```python
from strukt.middleware import Middleware
from strukt.types import InvocationState, HandlerResult

class LoggingMiddleware(Middleware):
    def before_handle(self, state: InvocationState, query_type: str, parts: list[str]):
        print(f"Before handling: {query_type}")
        return state, parts
    
    def after_handle(self, state: InvocationState, query_type: str, result: HandlerResult):
        print(f"After handling: {result.status}")
        return result

# Use with FastMCP server
server = StruktFastMCPServer(
    handlers=handlers,
    middleware=[LoggingMiddleware()]
)
```

### Authentication Middleware

API key authentication is automatically applied when configured:

```python
# Authentication is handled automatically when enabled in config
config.mcp_v2.auth_api_key.enabled = True
```

## Advanced Usage

### Custom Tool Registration

For more control over tool registration:

```python
from strukt.mcp_v2 import discover_mcp_methods, build_fastmcp_tool_from_method

# Discover tools from a handler
tools = discover_mcp_methods(handler, "my_handler")

# Manually register with FastMCP
for tool_spec in tools:
    tool_func = build_fastmcp_tool_from_method(tool_spec)
    server.mcp.add_tool(tool_func)
```

### Direct FastMCP Access

Access the underlying FastMCP instance for advanced features:

```python
# Get the FastMCP instance
fastmcp = server.get_fastmcp_instance()

# Use FastMCP features directly
@fastmcp.tool
def custom_tool(param: str) -> str:
    """Custom tool added directly to FastMCP."""
    return f"Custom: {param}"
```

### FastAPI Integration

```python
from strukt.mcp_v2 import build_fastapi_app

# Create a Strukt app (in real usage, this would be your existing app)
config = StruktConfig()
config.mcp_v2.enabled = True
config.mcp_v2.server_name = "struktx-api"
config.mcp_v2.include_handlers = ["example", "user"]

# Mock Strukt app for this example
class MockStrukt:
    def __init__(self):
        self._engine = type('Engine', (), {
            '_handlers': {
                "example": ExampleHandler(),
                "user": UserHandler()
            },
            '_middleware': []
        })()
    
    def get_memory(self):
        return None

strukt_app = MockStrukt()

# Create FastAPI app with MCP v2 integration
app = build_fastapi_app(strukt_app, config)

# The app now has MCP endpoints at /mcp
# - POST /mcp - JSON-RPC endpoint
# - GET /mcp/tools - List available tools
# - GET /mcp/health - Health check
```

### Standalone FastAPI App

```python
from strukt.mcp_v2 import create_mcp_v2_app

handlers = {
    "example": ExampleHandler(),
    "user": UserHandler()
}

config = StruktConfig()
config.mcp_v2.enabled = True
config.mcp_v2.server_name = "standalone-mcp"
config.mcp_v2.include_handlers = ["example", "user"]

# Create standalone app
app = create_mcp_v2_app(
    handlers=handlers,
    config=config,
    server_name="standalone-mcp"
)
```

### Error Handling

```python
def mcp_risky_operation(self, data: str) -> str:
    """A tool that might fail."""
    try:
        # Risky operation
        result = process_data(data)
        return result
    except Exception as e:
        # Errors are automatically handled by FastMCP
        raise ValueError(f"Operation failed: {e}")
```

## CLI Usage

### Command Line Interface

```bash
# Run with stdio transport (default)
python -m strukt.mcp_v2.cli_example

# Run with SSE transport
python -m strukt.mcp_v2.cli_example --transport sse --port 8000

# Run with configuration file
python -m strukt.mcp_v2.cli_example --config config.yaml

# Run with custom server name
python -m strukt.mcp_v2.cli_example --server-name "my-mcp-server"

# List available tools
python -m strukt.mcp_v2.cli_example --list-tools
```

### Configuration Migration

```bash
# Migrate your config file from v1 to v2
python -m strukt.mcp_v2.migrate_config config.yaml

# Migrate to new file
python -m strukt.mcp_v2.migrate_config config.yaml config_v2.yaml

# Migrate without backup
python -m strukt.mcp_v2.migrate_config config.yaml --no-backup
```

## Troubleshooting

### Common Issues

1. **Tool Not Discovered**:
   - Ensure method starts with `mcp_`
   - Check that handler is in `include_handlers` list
   - Verify method is callable (not a property)

2. **Schema Generation Issues**:
   - Use proper type hints
   - Avoid complex generic types
   - Use Pydantic models for complex objects

3. **Authentication Failures**:
   - Check API key is set in environment
   - Verify header name matches configuration
   - Ensure auth is enabled in config

4. **Transport Issues**:
   - Verify FastMCP version supports chosen transport
   - Check port availability for SSE/HTTP
   - Ensure proper network configuration

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("mcp_v2").setLevel(logging.DEBUG)
```

### Testing Tools

Test your tools independently:

```python
# Test tool discovery
from strukt.mcp_v2 import discover_mcp_methods
tools = discover_mcp_methods(handler, "my_handler")
print(f"Discovered {len(tools)} tools")

# Test individual tool
tool_spec = tools[0]
tool_func = build_fastmcp_tool_from_method(tool_spec)
result = tool_func(timezone="UTC")
print(f"Tool result: {result}")
```

## API Reference

### StruktFastMCPServer

Main server class for FastMCP v2 integration.

```python
class StruktFastMCPServer:
    def __init__(
        self,
        *,
        handlers: Dict[str, Handler],
        include_handlers: List[str] | None = None,
        server_name: str = "struktx-mcp",
        memory: MemoryEngine | None = None,
        config: MCPConfig | None = None,
        middleware: List[Any] | None = None,
    ) -> None
```

### Transport Functions

```python
def run_stdio(server: StruktFastMCPServer) -> None
def run_sse(server: StruktFastMCPServer, host: str = "localhost", port: int = 8000) -> None
def run_http(server: StruktFastMCPServer, host: str = "localhost", port: int = 8000) -> None
```

### Tool Discovery Functions

```python
def discover_mcp_methods(handler: Handler, handler_key: str) -> List[ToolSpec]
def build_fastmcp_tool_from_method(tool_spec: ToolSpec) -> Callable[..., Any]
```

### FastAPI Integration

```python
def build_fastapi_app(strukt_app: Strukt, cfg: StruktConfig, *, app: Any | None = None, prefix: str = "/mcp") -> Any
def create_mcp_v2_app(handlers: Dict[str, Any], config: StruktConfig, *, server_name: str | None = None, include_handlers: list[str] | None = None) -> Any
```

## Contributing

When contributing to the FastMCP v2 integration:

1. Maintain backward compatibility with existing Strukt patterns
2. Follow the convention-based approach for tool discovery
3. Ensure type safety and proper schema generation
4. Add comprehensive tests for new features
5. Update documentation for any API changes

## License

This module is part of StruktX and follows the same MIT license.