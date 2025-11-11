# Spelling Bee Environment Documentation

## Overview
**Spelling Bee** is a two-player word game where players take turns creating valid English words using only a limited set of allowed letters. The challenge increases as the game progresses because each new word must be at least as long as the previous word. Players must balance vocabulary knowledge with strategic thinking to outmaneuver their opponent by forcing them into increasingly difficult positions. This implementation features frequency-weighted letter selection to ensure balanced and playable letter sets.

## Action Space

- **Format:** Actions are strings representing a valid English word wrapped in square brackets.
- **Examples:**
  - Submit the word "example": `[example]`
  - Submit the word "fantastic": `[fantastic]`
- **Notes:** Words must only contain letters from the allowed set, must be at least as long as the previous word, and cannot be words that have already been used in the game.

## Observation Space

**Reset Observations**
On reset, each player receives a prompt containing the allowed letters for the game. For example:

```plaintext
You are Player 0 in the Spelling Bee Game.
Allowed Letters: aeinrst
Each word must be at least as long as the previous word.
Repeated words are not allowed.
Wrap your word in square brackets, e.g., '[example]'.
```

**Step Observations**
During gameplay, players can observe all words submitted so far. For example:

```plaintext
[Player 0] I'll start with a short word: [tar]
[Player 1] Building on that length: [nest]
[Player 0] I'll use more letters now: [strain]
[Player 1] This is getting challenging. Let me try: [transit]
```

## Gameplay

- **Players:** 2 players
- **Initial Setup:** A random set of unique letters is selected based on frequency in English
- **Turns:** Players take turns submitting valid words
- **Progression:** Each word must be at least as long as the previous word
- **Objective:** Force your opponent into a position where they cannot create a valid word

## Key Rules

1. **Letter Set:**
   - A fixed number of unique letters (default varies by variant) is randomly selected at the start
   - Letter selection is weighted by frequency in English to ensure playable sets
   - All words must be formed using only these allowed letters

2. **Word Requirements:**
   - Words must be valid English words (verified against a dictionary)
   - Words must contain only letters from the allowed set
   - Words must be at least as long as the previous word in the game
   - Words cannot be repeated within the same game

3. **Valid Moves:**
   - Players must submit their word in square brackets: `[word]`
   - The word must meet all the requirements above
   - Players may include additional text before or after their word submission

4. **Winning Conditions:**
   - **Win:** When the opponent cannot create a valid word that meets all requirements
   - **Loss:** When you cannot create a valid word that meets all requirements

5. **Game Termination:**
   - The game concludes when a player fails to submit a valid word
   - There is no predefined turn limit; the game continues until one player cannot make a valid move

## Rewards

| Outcome     | Reward for Winner | Reward for Loser |
|-------------|:-----------------:|:----------------:|
| **Win**     | `+1`              | `-1`             |
| **Invalid** | `-1`              | `0`              |

## Parameters

- `num_letters` (`int`):
  - **Description:** Sets the number of unique letters allowed in the game
  - **Impact:** More letters provide greater word-forming possibilities, while fewer letters increase difficulty

## Variants

| Env-id                 | num_letters |
|------------------------|:-----------:|
| `SpellingBee-v0`       | `7`         |
| `SpellingBee-v0-small` | `4`         |
| `SpellingBee-v0-large` | `10`        |


### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg
