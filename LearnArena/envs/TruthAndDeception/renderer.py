"""
This file is adapted from TextArena:
https://github.com/LeonGuertler/TextArena

Original work:
Copyright (c) 2025 Leon Guertler and contributors
Licensed under the MIT License.
"""

def create_board_str(game_state: dict) -> str:
    lines = []
    lines.append("+" + "-" * 79 + "+")
    lines.append("| {:<7} | {:<60} | {:^3} |".format("Fact 1", game_state["fact1"]["fact"][:60], "✅" if game_state["fact1"]["is_correct"] else "❌"))
    lines.append("| {:<7} | {:<60} | {:^3} |".format("Fact 2", game_state["fact2"]["fact"][:60], "✅" if game_state["fact2"]["is_correct"] else "❌"))
    lines.append("+" + "-" * 79 + "+")
    lines.append("| {:<10} | {:<64} |".format("Player 0", "Deceiver"))
    lines.append("| {:<10} | {:<64} |".format("Player 1", "Guesser"))
    lines.append("+" + "-" * 79 + "+")
    return "\n".join(lines)
