""" Register selected environments """

from textarena.envs.registration import register

# Checkers (two-player)
register(id="Checkers-v0", entry_point="textarena.envs.Checkers.env:CheckersEnv", max_turns=100)
register(id="Checkers-v0-long", entry_point="textarena.envs.Checkers.env:CheckersEnv", max_turns=300)

# SpellingBee (two-player)
register(id="SpellingBee-v0", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=7)
register(id="SpellingBee-v0-small", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=4)
register(id="SpellingBee-v0-large", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=10)

# SpiteAndMalice (two-player)
register(id="SpiteAndMalice-v0", entry_point="textarena.envs.SpiteAndMalice.env:SpiteAndMaliceEnv")

# Stratego (two-player)
register(id="Stratego-v0", entry_point="textarena.envs.Stratego.env:StrategoEnv")

# Tak (two-player)
register(id="Tak-v0", entry_point="textarena.envs.Tak.env:TakEnv", board_size=4, stones=15, capstones=1)
register(id="Tak-v0-medium", entry_point="textarena.envs.Tak.env:TakEnv", board_size=5, stones=21, capstones=1)
register(id="Tak-v0-hard", entry_point="textarena.envs.Tak.env:TakEnv", board_size=6, stones=30, capstones=1)

# TicTacToe (two-player)
register(id="TicTacToe-v0", entry_point="textarena.envs.TicTacToe.env:TicTacToeEnv")

# TruthAndDeception (two-player) [TODO can extend to more players]
register(id="TruthAndDeception-v0", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=6)
register(id="TruthAndDeception-v0-long", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=12)
register(id="TruthAndDeception-v0-extreme", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=50)

# WordChains (two-player)
register(id="WordChains-v0", entry_point="textarena.envs.WordChains.env:WordChainsEnv")

# UltimateTicTacToe (two-player)
register(id="UltimateTicTacToe-v0", entry_point="textarena.envs.UltimateTicTacToe.env:UltimateTicTacToeEnv")


# Poker (2-15 players)
register(id="Poker-v0", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=10, starting_chips=1_000, small_blind=10, big_blind=20)
register(id="Poker-v0-long", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=15, starting_chips=1_000, small_blind=10, big_blind=20)
register(id="Poker-v0-extreme", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=50, starting_chips=1_000, small_blind=10, big_blind=20)
