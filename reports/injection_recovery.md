# Injection-recovery test

300 fake transits in 50 quiet sector-40 stars; 21 fell in data gaps (not observable, left out); 0 errors.

### NEW pipeline

Each box: % CANDIDATE / % CANDIDATE+REVIEW  (n = observable fake transits)

| duration \ depth | 300 ppm | 600 ppm | 1000 ppm | 2000 ppm | 3500 ppm | 5000 ppm |
|---|---|---|---|---|---|---|
| 2 h | 0 / 0 (n=10) | 0 / 0 (n=9) | 30 / 60 (n=10) | 44 / 89 (n=9) | 70 / 100 (n=10) | 67 / 89 (n=9) |
| 4 h | 0 / 0 (n=10) | 10 / 10 (n=10) | 12 / 38 (n=8) | 67 / 67 (n=9) | 70 / 90 (n=10) | 50 / 100 (n=10) |
| 8 h | 0 / 0 (n=9) | 0 / 20 (n=10) | 33 / 44 (n=9) | 20 / 70 (n=10) | 40 / 80 (n=10) | 78 / 100 (n=9) |
| 12 h | 0 / 0 (n=9) | 10 / 20 (n=10) | 38 / 75 (n=8) | 25 / 75 (n=8) | 67 / 89 (n=9) | 40 / 70 (n=10) |
| 16 h | 10 / 20 (n=10) | 40 / 40 (n=10) | 10 / 30 (n=10) | 14 / 29 (n=7) | 33 / 67 (n=9) | 75 / 88 (n=8) |

Overall (279 observable): NOTHING 99 (35%), CANDIDATE 88 (32%), REVIEW 56 (20%), OTHER DIP 20 (7%), REJECT 16 (6%)

### OLD pipeline

Each box: % CANDIDATE / % CANDIDATE+REVIEW  (n = observable fake transits)

| duration \ depth | 300 ppm | 600 ppm | 1000 ppm | 2000 ppm | 3500 ppm | 5000 ppm |
|---|---|---|---|---|---|---|
| 2 h | 0 / 0 (n=10) | 0 / 0 (n=9) | 40 / 50 (n=10) | 56 / 78 (n=9) | 50 / 100 (n=10) | 44 / 89 (n=9) |
| 4 h | 0 / 0 (n=10) | 10 / 10 (n=10) | 25 / 50 (n=8) | 78 / 78 (n=9) | 70 / 90 (n=10) | 60 / 100 (n=10) |
| 8 h | 0 / 0 (n=9) | 10 / 20 (n=10) | 44 / 67 (n=9) | 40 / 70 (n=10) | 40 / 90 (n=10) | 78 / 100 (n=9) |
| 12 h | 0 / 0 (n=9) | 20 / 40 (n=10) | 62 / 88 (n=8) | 88 / 100 (n=8) | 89 / 100 (n=9) | 80 / 90 (n=10) |
| 16 h | 10 / 20 (n=10) | 40 / 40 (n=10) | 40 / 70 (n=10) | 43 / 57 (n=7) | 67 / 78 (n=9) | 88 / 88 (n=8) |

Overall (279 observable): CANDIDATE 116 (42%), NOTHING 108 (39%), REVIEW 46 (16%), OTHER DIP 9 (3%)

### What removed fake transits (new pipeline)

| Reason | Fake transits |
|---|---|
| too weak: nothing found | 99 |
| a different dip was stronger | 20 |
| flagged points inside the dip | 7 |
| changes with cleaning | 5 |
| flare next to the dip | 3 |
| background rises during the dip | 2 |
| same-time dips in other stars | 1 |

### Why fake transits ended up in REVIEW (new pipeline)

| Soft check | Fake transits |
|---|---|
| too big for a planet | 20 |
| thruster firing nearby | 13 |
| borderline SNR | 10 |
| manually excluded points inside the dip | 8 |
| same-time dips in some other stars | 8 |
| changes somewhat with cleaning | 7 |
| V-shaped | 5 |
| weaker in raw SAP flux | 3 |
| background rises a little | 1 |