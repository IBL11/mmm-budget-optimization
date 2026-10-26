# Team Workflow

## Branches
- `main` – final, reviewed work only (protected, no direct pushes)
- `dev` – integration branch; all feature branches merge here
- `feature/<task-name>` – one branch per task, e.g. `feature/data-cleaning`, `feature/optimizer`

## Daily workflow
```bash
git checkout dev
git pull origin dev
git checkout -b feature/<task-name>     # first time only
# ... work ...
git add .
git commit -m "Add outlier handling for spend columns"
git push origin feature/<task-name>
```
Then open a **Pull Request into `dev`**. The Team Lead reviews and merges.

## Rules
1. Never push directly to `main` or `dev`.
2. Pull `dev` before starting work each day.
3. Clear commit messages: what you changed, not "update".
4. Don't commit large data files (>50MB), venv folders or credentials.
5. Clear notebook outputs before committing if they're huge.
6. Keep column names exactly as defined in `config/data_schema.md`.
7. Save outputs for Power BI only in `outputs/tables/` using the agreed file names.

## Agreed output files
| File | Produced by | Used by |
|---|---|---|
| `data/processed/clean_data.csv` | Member 2 | Members 3, 4, 5, 6 |
| `data/processed/model_data.csv` | Member 3 | Members 5, 6 |
| `config/params.json` | Member 5 | Members 6, 7, 8 |
| `outputs/tables/model_metrics.csv` | Member 6 | Members 9, 10 |
| `outputs/tables/contributions.csv` | Member 7 | Members 9, 10 |
| `outputs/tables/roi_table.csv` | Member 7 | Members 8, 9, 10 |
| `outputs/tables/response_curves.csv` | Member 7 | Member 9 |
| `outputs/tables/external_impact.csv` | Member 3 | Members 9, 10 |
| `outputs/tables/optimal_allocation.csv` | Member 8 | Members 9, 10 |
| `outputs/tables/budget_scenarios.csv` | Member 8 | Members 9, 10 |
