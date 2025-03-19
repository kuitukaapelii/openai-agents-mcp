"""Tests for MCP agent hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from agents.lifecycle import AgentHooks

from agents_mcp.agent import Agent
from agents_mcp.agent_hooks import MCPAgentHooks


@pytest.fixture
def mock_agent():
    """Create a mock MCP agent."""
    agent = Agent(
        name="TestAgent",
        instructions="Test instructions",
        mcp_servers=["fetch", "filesystem"],
    )
    # Mock the load_mcp_tools method
    agent.load_mcp_tools = AsyncMock()
    return agent


@pytest.fixture
def mock_original_hooks():
    """Create mock original hooks."""
    hooks = MagicMock(spec=AgentHooks)
    hooks.on_start = AsyncMock()
    hooks.on_tool_call = AsyncMock()
    hooks.on_tool_result = AsyncMock()
    hooks.on_model_response = AsyncMock()
    hooks.on_agent_response = AsyncMock()
    hooks.on_agent_finish = AsyncMock()
    return hooks


@pytest.fixture
def mcp_hooks(mock_agent, mock_original_hooks):
    """Create MCP agent hooks with mocked agent and original hooks."""
    return MCPAgentHooks(agent=mock_agent, original_hooks=mock_original_hooks)


@pytest.mark.asyncio
async def test_mcp_hooks_initialization(mock_agent, mock_original_hooks):
    """Test initialization of MCPAgentHooks."""
    hooks = MCPAgentHooks(agent=mock_agent, original_hooks=mock_original_hooks)

    assert hooks.agent is mock_agent
    assert hooks.original_hooks is mock_original_hooks


@pytest.mark.asyncio
async def test_on_start_loads_mcp_tools(mcp_hooks, run_context_wrapper):
    """Test that on_start loads MCP tools."""
    # Call on_start
    await mcp_hooks.on_start(run_context_wrapper, mcp_hooks.agent)

    # Verify load_mcp_tools was called
    mcp_hooks.agent.load_mcp_tools.assert_called_once_with(run_context_wrapper)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_start.assert_called_once_with(run_context_wrapper, mcp_hooks.agent)


@pytest.mark.asyncio
async def test_on_tool_call_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_tool_call calls the original hook."""
    # Call on_tool_call
    call = {"type": "function", "function": {"name": "test_tool"}}
    await mcp_hooks.on_tool_call(run_context_wrapper, call, 0)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_tool_call.assert_called_once_with(run_context_wrapper, call, 0)


@pytest.mark.asyncio
async def test_on_tool_result_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_tool_result calls the original hook."""
    # Call on_tool_result
    result = {"name": "test_tool", "result": "test result"}
    await mcp_hooks.on_tool_result(run_context_wrapper, result, 0)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_tool_result.assert_called_once_with(run_context_wrapper, result, 0)


@pytest.mark.asyncio
async def test_on_model_response_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_model_response calls the original hook."""
    # Call on_model_response
    response = {"content": "test response"}
    await mcp_hooks.on_model_response(run_context_wrapper, response)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_model_response.assert_called_once_with(
        run_context_wrapper, response
    )


@pytest.mark.asyncio
async def test_on_agent_response_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_agent_response calls the original hook."""
    # Call on_agent_response
    response = {"content": "test response"}
    await mcp_hooks.on_agent_response(run_context_wrapper, response)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_agent_response.assert_called_once_with(
        run_context_wrapper, response
    )


@pytest.mark.asyncio
async def test_on_agent_finish_calls_original_hook(mcp_hooks, run_context_wrapper):
    """Test that on_agent_finish calls the original hook."""
    # Call on_agent_finish
    output = {"final_response": "test output"}
    await mcp_hooks.on_agent_finish(run_context_wrapper, output)

    # Verify original hook was called
    mcp_hooks.original_hooks.on_agent_finish.assert_called_once_with(run_context_wrapper, output)
