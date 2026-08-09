"""Test doubles that stand in for external services.

These are committed rather than mocked because the thing under test is the plumbing --
base_url routing, error handling, the fallback chain, and how the abstraction copes with an
endpoint that does not behave like OpenAI. A mock asserts what we believed; a server asserts
what the client library actually does over HTTP.
"""
