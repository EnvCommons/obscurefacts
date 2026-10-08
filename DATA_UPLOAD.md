# Data Upload Requirements for ObscureFacts

## Overview
This environment requires the following data to be uploaded to OpenReward cloud storage.

## Directory Structure
```
/orwd_data/
└── tasks.json (Size: ~10 KB)
```

## File Descriptions
- **tasks.json**: JSON array containing 49 trivia questions with id, question, and answer fields

## Upload Instructions
Upload `tasks.json` to the root of the data bucket of the OpenReward environment `GeneralReasoning/ObscureFacts`, so that it is mounted at `/orwd_data/tasks.json` (the path `obscurefacts.py` reads).
