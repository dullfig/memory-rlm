@echo off
REM MCP server launcher — ensures binary exists, then starts the server
call "%~dp0..\scripts\bootstrap.cmd"
"%~dp0memory-rlm.exe" serve
