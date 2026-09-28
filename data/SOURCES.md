# Data sources

The project uses the 104-match FIFA World Cup 2026 JSON snapshot from the OpenFootball `worldcup.json` project, consistent with Assessment 2.

- Repository: https://github.com/openfootball/worldcup.json
- Intended file: `2026/worldcup.json`
- Accessed for the course project in August 2026

Before modelling, the team will preserve a source snapshot in `data/raw/`, record its checksum, and validate the row count, match dates, teams, scores, and required fields. The modelling tables will be built in chronological order. Features for a match will use only results available before that match date.

