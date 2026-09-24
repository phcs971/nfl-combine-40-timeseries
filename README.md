# NFL Combine 40-Yard Dash — Position-Time Series

Position vs. time series for NFL Combine 40-yard dashes (2026 combine), sampled to
represent three athlete classes rather than the fastest athletes in each.

## Athlete classes

| Class | Positions |
|---|---|
| `skill` | WR, CB, SAF |
| `strong` | RB, TE, LB, EDGE |
| `line` | OT, G, C, DT |

## Video corpus

11 NFL-channel YouTube videos (`data/videos.csv`), all 30 fps, one per position group.
Videos were uploaded within days of the 2026 combine.

| video_id | Group | Positions | Class |
|---|---|---|---|
| 0r_usy_GIAI | Wide Receivers Group 1 | WR | skill |
| BEFga72DF3U | Wide Receivers Group 2 | WR | skill |
| -zgIolzsJxY | Defensive Backs | CB | skill |
| UcatZ1FYe5E | Safeties | SAF | skill |
| g__zmhX838U | Running Backs | RB | strong |
| 9xc35LbeeUE | Tight Ends | TE | strong |
| olzS7RidilY | Linebackers | LB | strong |
| LDqozfzjaNU | Edge Rushers | EDGE | strong |
| W23Nvg3rw6U | Offensive Lineman Group 1 | OT\|G\|C | line |
| f9RAoGpGkIY | Offensive Lineman Group 2 | OT\|G\|C | line |
| Sr-Q6UjJq6g | Defensive Line | DT | line |

Each video shows a full position-group session: athlete name and bib on screen
continuously, a live 40-yd clock (0.01 s resolution) that starts on first movement
and freezes at the finish, and numbered distance mats on the lane.

## Data & licensing

No video or extracted frames are committed — footage is NFL-copyrighted. This repo
ships URLs and derived numeric measurements only. Combine and draft data from
[nflverse](https://github.com/nflverse/nflverse-data).

Code MIT. Derived measurements CC BY 4.0.
