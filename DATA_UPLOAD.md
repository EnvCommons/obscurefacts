# Data Upload Requirements for ObscureFacts

## Overview
This environment requires the following data to be uploaded to OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── tasks.json (Size: ~10 KB)
```

## File Descriptions
- **tasks.json**: JSON array containing 50 trivia questions with id, question, and answer fields

## Upload Instructions
Upload `tasks.json` to the root of the file store of the OpenReward environment `GeneralReasoning/ObscureFacts`, so that it is mounted at `/orwd_data/tasks.json` (the path `obscurefacts.py` reads). From this directory:

```
orwd upload GeneralReasoning/ObscureFacts tasks.json
orwd files GeneralReasoning/ObscureFacts
```

`orwd files` should list a single `tasks.json` with the new size and timestamp. The deployed server reads only `/orwd_data/tasks.json`; the copy in this repository is used for local development and tests, so keep the two identical. Changing the repository copy alone changes nothing in production.

`obscurefacts.py` loads `tasks.json` once, when the server starts, so a running server keeps serving the old questions and answers after an upload. Redeploy (push a commit to `main`) once the upload has finished so every replica loads the new file.

Training runs pin the task count and address tasks by position (`task_train_<i>` is entry `i`). While a run depends on the list, edit entries in place only: removing, adding or reordering a task shifts every later index, and the last index then fails with "400 Invalid index". Make structural changes only when a run restarts with the count re-pinned.
