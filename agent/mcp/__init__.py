"""MCP (Model Context Protocol) local servers.

This package contains local MCP server implementations for the Forge Agent.
These servers run as local processes (not Docker containers) and provide
tools via the MCP protocol.

Note: The filesystem MCP server has been migrated to use the official
Docker image (mcp/filesystem) instead of a local implementation.
"""
