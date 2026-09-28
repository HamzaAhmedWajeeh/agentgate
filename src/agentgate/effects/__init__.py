"""Actions past the gate: what the drafter may propose, and where an approved effect goes.

The drafter proposes; it holds no tool that can perform. The gate shows the human exactly the
proposals, and records a hash of what it showed. ``execute`` performs only proposals whose hash
matches what was approved, each through the effect sink -- and the only sink is an append-only
outbox, so nothing real can happen. Leak inventory item 24 is why this package exists: until it
did, the gate approved a draft and there was no action for it to approve.
"""
