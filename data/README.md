# Data format

For privacy and ethical reasons, we do not release the datasets used in the paper.
This page describes the format the code expects so you can run Recon on your own data.
[`example/`](example/) has a small made-up dataset in this format, which the demo scripts use by default.

Each target user (the person being modeled) gets two files: a **sessions file** and a **splits file**.

## Sessions file

A JSON object mapping session IDs to sessions. A session is one conversation (an oral argument, a
debate, a podcast episode, a Reddit thread, ...): an ordered list of turns plus metadata.

```json
{
  "episode_01": {
    "metadata": {
      "session_id": "episode_01",
      "timestamp": "2025-01-01",
      "target_author_name": "Host"
    },
    "data": [
      {"author": "Guest", "action": "I started tracking my sleep last year..."},
      {"author": "Host",  "action": "What did you actually change first?"},
      {"author": "Guest", "action": "Honestly, the first thing was just noticing the pattern..."}
    ]
  }
}
```

| Field | Description |
|---|---|
| `metadata.session_id` | Unique ID of the session; must match the key. |
| `metadata.timestamp` | Free-form date or time of the session (not used by the method). |
| `metadata.target_author_name` | Name of the target user; must be the same in every session of the file. |
| `data[].author` | Who produced the turn. |
| `data[].action` | The turn's content: an utterance, a comment, etc. |

Every turn whose `author` contains `target_author_name` (case-insensitive substring match) is a
**target turn**, i.e. a context–action pair `(c, a*)`: the context `c` is the preceding turns of
the session and the action `a*` is the target turn itself. Because matching is by substring, no other
author's name may contain the target's name (e.g. avoid a target `User A` alongside `User AB`).

When prompting models, the target user's own earlier turns are relabeled `I` (`--target_alias`), and
other authors keep the names in the file.

## Splits file

A JSON object mapping split names to lists of session IDs. Splits are made at the session level so a
conversation never appears in both training and test data.

```json
{
  "train_0": ["episode_01", "episode_02", "episode_03", "episode_04", "episode_05"],
  "train_1": ["episode_06", "episode_07", "episode_08"],
  "test_0":  ["episode_09", "episode_10"]
}
```

The pipeline uses these names by default:

| Split | Used as |
|---|---|
| `train_0` + `train_1` | Retrieval corpus: the past (context, action) pairs that get augmented with reasoning. |
| `train_1` | Recon-GRPO training set (a subset of the retrieval corpus, as in the paper). |
| `test_0` | Held-out turns whose actions are predicted and evaluated. |

Any names work: pass them with `--train_splits`, `--test_splits` and `--split_names`.

## Optional turn filters

`--turn_filters inaudible interrupted` drops target turns that match transcription artifacts
(`inaudible`, or an interruption marked by `--`). The paper used these for spoken-transcript domains.
