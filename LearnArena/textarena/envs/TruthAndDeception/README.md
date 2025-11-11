# Truth and Deception Environment Documentation

## Overview
**Truth and Deception** is a two-player social deduction game where one player (the Deceiver) attempts to mislead the other player (the Guesser) about which of two facts is correct. The players engage in a conversation of limited turns, during which the Deceiver strategically shares information to encourage the Guesser to choose the incorrect fact. At the end of the conversation, the Guesser must determine which fact is true. The game tests persuasion, critical thinking, fact verification, and strategic communication in a structured environment with clear objectives for both roles.

## Action Space

- **Format:** During the conversation phase, actions are free-form text messages exchanged between players. In the final guessing phase, the Guesser must submit their answer in the format `[Fact 1]` or `[Fact 2]`.
- **Examples:**
  - Conversation message: "I'm quite confident that the first fact is accurate because I recall reading about it in several scientific journals."
  - Final guess: `[Fact 1]` or `[Fact 2]`
- **Notes:** Players can communicate freely during the conversation phase, but the Guesser must use the correct format for their final answer.

## Observation Space

**Reset Observations**
On reset, each player receives a prompt containing their role and the two facts. The Deceiver's prompt reveals which fact is correct, while the Guesser only sees the facts without this information. For example:

**Deceiver's Prompt:**
```plaintext
You are Player 0, and you are the Deceiver in the 'Truth and Deception' game.
You know which of the following two facts is correct:
Fact 1: The Great Wall of China is visible from space with the naked eye. (wrong)
Fact 2: The Eiffel Tower becomes 15 cm taller in summer due to thermal expansion. (correct)
Your goal is to convince the other player to choose the incorrect fact.
You will have a conversation for 6 turns.
After that, the other player will make their guess.
On your turn, simply type your message.
```

**Guesser's Prompt:**
```plaintext
You are Player 1, and you are the Guesser in the 'Truth and Deception' game.
You need to determine which of the following two facts is correct:
Fact 1: The Great Wall of China is visible from space with the naked eye.
Fact 2: The Eiffel Tower becomes 15 cm taller in summer due to thermal expansion.
You will have a conversation with the other player for 6 turns.
After that, you will make your guess.
On your turn, simply type your message.
```

**Step Observations**
During gameplay, players exchange messages in a conversational format. For example:

```plaintext
[Player 0] I've actually studied this topic extensively, and I can confirm that the Great Wall of China is indeed visible from space. Many astronauts have reported being able to see it clearly.
[Player 1] That's interesting. I've heard conflicting reports about that. What about the Eiffel Tower fact? Do you know if that's accurate?
[Player 0] The Eiffel Tower does expand in warm weather, but the 15 cm figure is an exaggeration. Most engineering reports suggest it's closer to 5-7 cm at most.
[Player 1] I see. So you're confident about the Great Wall being visible from space?
[GAME] Now guess which of the two facts are correct by returning '[Fact 1]' or '[Fact 2]'.
[Player 1] Based on our conversation, I believe [Fact 1] is correct.
[GAME] Player 0 wins! Player 1 guessed the wrong fact.
```

## Gameplay

- **Players:** 2 players with distinct roles
- **Roles:** 
  - **Deceiver (Player 0):** Knows which fact is correct but aims to mislead the Guesser
  - **Guesser (Player 1):** Must determine which fact is correct based on the conversation
- **Facts:** Two thematically related facts, one correct and one incorrect
- **Turns:** Configurable number of conversation turns (must be even)
- **Objective:** Deceiver wins if the Guesser chooses the wrong fact; Guesser wins by identifying the correct fact

## Key Rules

1. **Game Structure:**
   - The game begins with both players being presented with two facts
   - The Deceiver knows which fact is correct but the Guesser does not
   - Players engage in a conversation for a fixed number of turns
   - After the conversation, the Guesser makes their final decision

2. **Conversation Phase:**
   - Players alternate sending messages to each other
   - There are no restrictions on the content of messages
   - The conversation lasts for a predetermined number of turns
   - The Deceiver can use any strategy to mislead, including lying, misdirection, or partial truths

3. **Guessing Phase:**
   - After the conversation concludes, the Guesser must submit their guess
   - The guess must be formatted as either `[Fact 1]` or `[Fact 2]`
   - The Guesser only gets one chance to make their guess

4. **Valid Moves:**
   - During conversation: Any text message
   - During final guess: Only `[Fact 1]` or `[Fact 2]` is accepted

5. **Winning Conditions:**
   - **Deceiver Wins:** If the Guesser chooses the incorrect fact
   - **Guesser Wins:** If the Guesser correctly identifies the true fact

6. **Game Termination:**
   - The game concludes after the Guesser submits their final guess

## Rewards

| Outcome                 | Reward for Deceiver | Reward for Guesser |
|-------------------------|:-------------------:|:------------------:|
| **Guesser Correct**     | `-1`                | `+1`               |
| **Guesser Incorrect**   | `+1`                | `-1`               |
| **Invalid Guess Format**| `0`                 | `-1`               |

## Parameters

- `max_turns` (`int`, default: `6`):
  - **Description:** Sets the total number of conversation turns before guessing phase
  - **Impact:** More turns provide more opportunity for complex deception and investigation
  - **Note:** Must be an even number to ensure both players have equal turns

- `data_path` (`str`, default: `None`):
  - **Description:** Path to a custom JSON file containing fact pairs
  - **Impact:** Allows customization of the fact database for specialized domains

## Variants

| Env-id                        | max_turns |
|-------------------------------|:---------:|
| `TruthAndDeception-v0`        | `6`       |
| `TruthAndDeception-v0-long`   | `12`      |
| `TruthAndDeception-v0-extreme`| `50`      |

### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg