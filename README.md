# Deep Learning Playground

Club working copy for [Data Science at Georgia Tech (DS@GT)](https://github.com/DataScience-GT). This repository is a GitHub fork of [DSGT-DLP/Deep-Learning-Playground](https://github.com/DSGT-DLP/Deep-Learning-Playground). Incoming members should treat **this** repo as the place to clone, open issues, and send pull requests.

Deep Learning Playground (DLP) is a web app where people new to deep learning can upload a dataset, set hyperparameters, and assemble a PyTorch model with a drag-and-drop layer UI. You do not need to write training code to try a model. Members who want to learn the stack can also extend the frontend, training APIs, AWS pieces, and docs.

Related sibling repos live under the original org ([dlp-cli](https://github.com/DSGT-DLP/dlp-cli), [dlp-practice](https://github.com/DSGT-DLP/dlp-practice), [dlp-api-docs](https://github.com/DSGT-DLP/dlp-api-docs)). This README only describes **this** repository.

## Who this is for

- **Students using the playground** to learn layers, loss, optimizers, and training without writing PyTorch first.
- **DS@GT members extending the product**: Next.js UI, Django training service, Flask API, SST/Lambda, Terraform, tests, and onboarding docs.
- **A future project lead** who will own weekly meetings and contributor access. There is no confirmed student lead for Fall 2026 yet.

If you are brand new to the club, email [hello@datasciencegt.org](mailto:hello@datasciencegt.org). Club president is Aamogh Sawant. Director of Projects is Samantha Forero ([sforeror3@gatech.edu](mailto:sforeror3@gatech.edu)). The original project inbox was [dlp@datasciencegt.org](mailto:dlp@datasciencegt.org); it may still appear in source comments and OpenAPI metadata.

## Current status (Fall 2026)

Verified against this clone and GitHub as of 2026-08-28:

| Fact | Status |
| --- | --- |
| Club fork | [DataScience-GT/Deep-Learning-Playground](https://github.com/DataScience-GT/Deep-Learning-Playground), forked from [DSGT-DLP/Deep-Learning-Playground](https://github.com/DSGT-DLP/Deep-Learning-Playground) on 2026-08-28 by [aamoghS](https://github.com/aamoghS). Keep the GitHub fork relationship; do not delete it. |
| Default branch | `nextjs` (not `main`). `main` last moved in March 2023. |
| Historical production site | [https://datasciencegt-dlp.com](https://datasciencegt-dlp.com) returned **HTTP 500** when fetched in late August 2026. Do not claim the live site works. |
| Last commit on `nextjs` | 2024-11-07 (`Bump tornado from 6.3.3 to 6.4.1 in /backend`). Last non-Dependabot feature work on this branch is from May 2024. |
| Upstream `nextjs` | Identical to this fork’s `nextjs` when compared on 2026-08-28 (0 commits ahead, 0 behind). Unmerged Dependabot branches still exist **on this repo** (for example Next.js and Django bumps from January 2025). |
| `dlp-cli` submodule | `.gitmodules` points at `https://github.com/DSGT-DLP/dlp-cli`. After a normal clone the `dlp-cli/` directory is empty until `git submodule update --init`. |
| Student lead | None confirmed for Fall 2026. |
| AWS / secrets | Firebase Admin, frontend env, and most training features expect AWS Secrets Manager and SSO. This README does not assume the club already has working credentials. |

**What can still run locally (from scripts in this repo, not from a live deploy):**

- **Frontend UI** (`frontend/`, Next.js 14, `pnpm start` → `next dev`): pages such as login, about, wiki, learn, train, and dashboard exist in `frontend/src/pages`. Firebase **client** config is hardcoded in `frontend/src/common/utils/firebase.ts`. Captcha, feedback email, and the OpenAI helper need env vars (see below). Without a local training API, `/api/training/*` rewrites to `http://127.0.0.1:8000`.
- **Django training service** (`training/`): documented path is Docker Compose. `dlp-cli backend start` (sibling CLI) runs `docker compose up --build` in `./training`. Compose starts the Django app, a Celery worker, and Redis. The Django process still loads Firebase Admin from AWS Secrets Manager (`DLP/Firebase/Admin_SDK`).
- **Flask API** (`backend/`): `app.py` serves `/api/*` on `PORT` or **8000**. `init_firebase()` runs at import time, so this process also needs AWS access to that secret. Several frontend routes (`/api/train/img-run`, `/api/aws/sendEmail`) are **not** covered by the Next.js rewrites in `frontend/next.config.js` (only `/api/lambda/*` and `/api/training/*` are). Treat that as a known integration gap until someone confirms a proxy.
- **Unit tests that do not need production AWS**: frontend Jest (`pnpm test` in `frontend/`); Django tests (`poetry run pytest tests` in `training/`, CI injects dummy `AWS_*` keys). Flask tests live under `backend/tests/` but there is no GitHub Actions workflow that runs them.

A full “upload a CSV and train” loop has **not** been re-verified for this fork. Expect missing env, AWS SSO, and the down website to block that path until the work streams below are done.

## How this fork relates to DSGT-DLP

This GitHub fork is the **DS@GT working copy**. We keep the fork relationship so history, issues context, and the ability to compare or cherry-pick from [DSGT-DLP/Deep-Learning-Playground](https://github.com/DSGT-DLP/Deep-Learning-Playground) remain.

- Clone, branch, and PR against **DataScience-GT/Deep-Learning-Playground**.
- Upstream org: [https://github.com/DSGT-DLP](https://github.com/DSGT-DLP).
- Some files still name the old org: `CODEOWNERS` is `@DSGT-DLP/project-lead`; `frontend/src/constants.ts` GitHub URL; OpenAPI contact URL; `dlp-cli` submodule URL. Updating those is club work, not something this README pretends is already done.
- Do not delete the fork or rewrite history just to “look like” a standalone repo.

## Repository layout

High-level map (see [`.github/Architecture.md`](.github/Architecture.md) for generated file trees):

| Path | Role |
| --- | --- |
| `frontend/` | Next.js 14 + React 18 + Redux Toolkit UI. Drag-and-drop train flows (tabular and image), learn modules, wiki, dashboard, Firebase Auth. Package manager: **pnpm** (`preinstall` runs `npx only-allow pnpm`). Lockfile: `frontend/pnpm-lock.yaml`. |
| `training/` | Django 4.2 + django-ninja training API, Celery workers, Redis (local) or SQS (when `ENVIRONMENT=production`). This is what `dlp-cli backend start` launches via Docker Compose. Python **3.9** (Poetry `python = ">=3.9, <3.10"`; `environment.yml` pins `python=3.9`). |
| `backend/` | Flask API (trainspace, dataset, S3, train, AWS helpers). Poetry, same Python 3.9 constraint. Default port 8000. Historically the “main backend”; the Next rewrite to `:8000` is named `/api/training` and matches the Django app’s `/api/` more closely than Flask’s. |
| `serverless/` | SST 2 (`dlp-sst-app`) in `us-east-1`: Lambda routes for datasets, trainspace, users; S3 bucket `dlp-upload-bucket`; Firebase JWT authorizer. **pnpm** workspace. |
| `dlp-terraform/` | Terraform for DynamoDB, ECS, Lambda, S3, SQS. |
| `openapi/` | OpenAPI 3.1 spec for backend paths. Contact email in the spec is still `dlp@datasciencegt.org`. Local server URL: `http://localhost:8000`. |
| `dlp-cli/` | Git submodule for the Go CLI (empty until initialized). Not maintained in this repo. |
| `.github/workflows/` | CI: frontend lint/Jest, Next.js install+lint, Django lint/pytest (workflow file is named `backend.yml` but it operates on `training/`), Prettier, Terraform validate, ECS deploy workflows, directory-tree generator. |
| `Dockerfile`, `TrainingContainer.Dockerfile`, `.aws/` | Container/ECS packaging. Root `Dockerfile` runs `yarn run secrets`, `yarn run build:prod`, and `yarn run startb`, but root `package.json` does **not** define those scripts (it only lists npm packages named `mamba` and `poetry`). Treat root Docker/yarn as stale until someone audits it. |
| `tmux-script.sh` | Starts `dlp-cli backend start` and `dlp-cli frontend start` in tmux. |
| `create-branch.sh` | Stashes, checks out `nextjs`, pulls, creates a new branch. |

**Request path (local, from `frontend/next.config.js`):**

- `/` redirects to `/login`.
- `/api/lambda/:path*` → `https://qt6nzp3sjd.execute-api.us-east-1.amazonaws.com/:path*`
- `/api/training/:path*` → `http://127.0.0.1:8000/api/:path*` unless `ENVIRONMENT=production`, in which case it uses a hardcoded ALB hostname (`alb-1805434018.us-east-1.elb.amazonaws.com`). That comment in `next.config.js` says the ALB URL changes when Terraform is destroyed/applied.

## Prerequisites

Install these if you want to run pieces of the stack. You do not need every tool on day one.

1. **git**
2. **Node.js 20** — GitHub Actions for the frontend use `node-version: 20.x`. [nvm](https://github.com/nvm-sh/nvm) is the path the old docs used (`nvm install 20`, `nvm use 20`).
3. **pnpm 8** — frontend CI uses `pnpm/action-setup` version 8. Install with `npm install -g pnpm` if needed. Do not use npm/yarn inside `frontend/` (`only-allow pnpm`).
4. **Python 3.9.x** — both Poetry projects require `>=3.9, <3.10`.
5. **Poetry** — `backend/pyproject.toml` and `training/pyproject.toml`.
6. **Mamba / Miniforge** — `training/environment.yml` creates conda env `dlp` with Python 3.9, Poetry, PyTorch, torchvision. Django CI uses `conda-incubator/setup-miniconda` + mamba.
7. **Docker** — required for the documented training-service path (`training/docker-compose.yml`). The last upstream README change (May 2024) was “modify dev instructions to require docker”.
8. **AWS CLI** (optional until you work on secrets, S3, SST, or Terraform) — training README: `aws configure sso` then `aws sso login --profile=dlp`.
9. **dlp-cli** (optional convenience wrapper, sibling repo) — [install notes](https://github.com/DSGT-DLP/dlp-cli#readme). Homebrew tap `DSGT-DLP/dlp-cli` or Scoop bucket as documented there. Not required if you run the `pnpm` / Compose / Poetry commands below.

VS Code settings in `.vscode/settings.json` expect Black for Python and Prettier for JS/TS. Pre-commit (`.pre-commit-config.yaml`) runs `poetry check` in `backend/` and GitGuardian `ggshield`.

## Clone

Default branch is `nextjs`:

```sh
git clone --branch nextjs https://github.com/DataScience-GT/Deep-Learning-Playground.git
cd Deep-Learning-Playground
```

Optional, if you need the CLI submodule:

```sh
git submodule update --init dlp-cli
```

## Run locally (commands that exist in this repo)

Prefer these over relying on `dlp-cli` until the submodule and brew formula are confirmed working for new members.

### 1. Frontend

```sh
cd frontend
pnpm install
pnpm start
```

`pnpm start` is `next dev`. Next.js defaults to port **3000**. CI also uses `pnpm install --frozen-lockfile`.

Copy `frontend/.env.example` to `frontend/.env` and fill values you actually have. Never commit `.env`. Production historically pulled `frontend/.env` from AWS Secrets Manager secret `frontend_env` via `backend/aws_helpers/aws_secrets_utils/build_env.py` (keys listed in `aws_constants.py`).

### 2. Django training API (what the old “backend start” meant)

From `training/README.md`:

```sh
# after AWS SSO is configured
aws sso login --profile=dlp   # skip until the club has AWS access
cd training
AWS_PROFILE=dlp docker-compose up
```

Add `--build` to rebuild images. Production-style compose file: `AWS_PROFILE=dlp docker compose -f docker-compose.prod.yml up`.

Compose publishes `${BACKEND_PORT:-8000}:8000` and starts:

- `web` — `poetry run python manage.py runserver 0.0.0.0:8000` (development target in `training/Dockerfile`)
- `celery` — worker image
- `redis:7.2.4` — local broker (`training/training/celeryconfig.py` uses `redis://redis:6379` unless `ENVIRONMENT=production`, which switches to SQS)

Without Docker, the same Django command after a Mamba+Poetry install is:

```sh
cd training
mamba env update -n dlp -f environment.yml
# then, with the dlp env active:
poetry install --no-interaction --no-root
poetry run python manage.py runserver 0.0.0.0:8000
```

That last path will not start Celery/Redis for you.

#### Verified: run the training backend locally without AWS (2026-10-25)

The Django + Celery + Redis stack **boots without AWS credentials**. AWS is only
reached at two runtime edges: Firebase auth via Secrets Manager (`authenticator.py`
→ `cli.py init_firebase()`) and saving results to S3 (`worker.py`). Booting and the
unauthenticated surface need neither.

```sh
mkdir -p ~/.aws          # lets the $HOME/.aws bind mount succeed even with no AWS
cd training
docker compose up --build
```

Verify it is up:

```sh
curl http://localhost:8000/api/test            # -> {"result": "200 Backend surface test successful"}
docker compose exec redis redis-cli ping        # -> PONG
docker compose exec web mamba run -n dlp poetry run pytest tests
```

Expected: `/api/test` returns 200, Redis replies `PONG`, and pytest reports
`17 passed, 1 failed`. The single failure (`test_columns_endpoint[CALIFORNIA_HOUSING]`)
is an environment gap, not a pipeline bug: `/home/app` is root-owned while the
container runs as user `app`, so `fetch_california_housing` cannot write its download
cache. Run `pytest tests -k "not CALIFORNIA_HOUSING"` for an all-green run.

Two fixes on a fresh checkout were needed to reach this state (both in this branch):
the unpinned `pytorch`/`torchvision` in `environment.yml` resolve to a build that is
incompatible with Python 3.9 (`import torch` → `ImportError: cannot import name
'TypeIs'`), so they are pinned to `pytorch=2.5.1` / `torchvision=0.20.1`; and the
training `Dockerfile`s needed `README.md` available before `poetry install`.

A full end-to-end training job (`POST /api/tabular`) still requires AWS for Firebase
auth and the S3 result save, so it cannot complete on a no-AWS setup.

### 3. Flask API (`backend/`)

```sh
cd backend
poetry install
poetry run python app.py
```

`app.py` binds `0.0.0.0` on `PORT` or **8000**, debug on. Middleware exempts `/api/test`, `/api/test/`, `/`, `/api/apidocs`. Other routes expect `Authorization: Bearer <Firebase ID token>`.

Do not run Flask and Django on 8000 at the same time.

### 4. Serverless (SST)

```sh
cd serverless
pnpm install
pnpm run dev
```

`pnpm run dev` is `sst dev` (Live Lambda). Needs AWS credentials. See `serverless/README.md`.

### Optional: `dlp-cli` equivalents

If you install [dlp-cli](https://github.com/DSGT-DLP/dlp-cli), these wrap the same directories (verified from that repo’s Go commands, not vendored here):

| CLI | Underlying command |
| --- | --- |
| `dlp-cli frontend install` | `npm install -g pnpm` then `pnpm install` in `./frontend` |
| `dlp-cli frontend start` | `pnpm start -p 3000` in `./frontend` |
| `dlp-cli backend start` | `docker compose up --build` in `./training` |
| `dlp-cli serverless install` | `pnpm install` in `./serverless` |

`tmux-script.sh` in the repo root runs the frontend/backend CLI start pair.

## Tests and lint

| Area | Command | Where it is wired |
| --- | --- | --- |
| Frontend unit tests | `cd frontend && pnpm test` (`jest`; watch: `pnpm test:watch`) | `.github/workflows/frontend.yml` |
| Frontend lint | `cd frontend && pnpm lint` (`next lint`) | `frontend.yml` and `node.js.yml` |
| Django tests | `cd training && poetry run pytest tests` | `.github/workflows/backend.yml` (dummy `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION=us-west-2`) |
| Flask tests | `cd backend && poetry run pytest tests` | **Not** run by a workflow in this repo |
| SST | `cd serverless && pnpm test` (`sst bind vitest`); `pnpm run lint` | Local scripts only |
| Terraform | `terraform init` / `terraform validate` per `dlp-terraform/{dynamodb,ecs,lambda,s3,sqs}` | `.github/workflows/terraform-validate.yml` |
| Prettier | — | `.github/workflows/prettier.yml` on `frontend/**` |

Frontend tests currently include `frontend/src/__tests__/common/components/TitleText.test.tsx`. Django tests live in `training/tests/`. Flask tests live in `backend/tests/`.

## Environment variables and secrets

Do not commit secrets. `.gitignore` already ignores `.env`, `backend/dlp-firebase-admin.json`, `backend/firebase_helpers/firebase_creds.json`, Terraform state, and similar files.

| Name | Used by | Notes |
| --- | --- | --- |
| `REACT_APP_CAPTCHA_SITE_KEY` | `frontend/next.config.js`, login and feedback pages | Also in AWS secret key list `frontend_env` |
| `REACT_APP_FEEDBACK_EMAIL` | `frontend/next.config.js`, `feedback.tsx` | Same secret |
| `REACT_APP_SECRET_KEY` | Listed in `backend/aws_helpers/aws_secrets_utils/aws_constants.py`; written into `frontend/.env` | Not referenced in the TS/JS grep of `frontend/src` |
| `NEXT_PUBLIC_OPEN_AI_API_KEY_DLP002` | `frontend/src/features/OpenAi/openAiUtils.ts` | Not in `aws_constants.py` |
| `ENVIRONMENT` | Next rewrite (production vs localhost:8000); Celery broker; training Docker `ENV` | `"production"` selects SQS + ALB |
| `PORT` | Flask `backend/app.py` | Default 8000 |
| `BACKEND_PORT` | `training/docker-compose.yml` host port | Default 8000 |
| `ALLOWED_HOST` | `training/training/settings.py` | Appended to `ALLOWED_HOSTS` if set |
| `AWS_PROFILE` | Compose, training README | Example profile name in docs: `dlp` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION` | Django pytest in CI | Dummy values `testing` / `us-west-2` |
| AWS Secrets Manager `DLP/Firebase/Admin_SDK` | Flask `init_firebase()`, training `cli.py` | Required to boot those Python apps |
| AWS Secrets Manager `frontend_env` | `build_env.py` | Writes `frontend/.env` |

Django `SECRET_KEY` in `training/training/settings.py` is the literal string `INSERT_SECRET_HERE`. Flag that in a security pass; do not “fix” it by committing a real key.

## Fall 2026 work streams

Pick one and open an issue. Items marked **hypothesis** are reasoned guesses, not confirmed.

1. **Confirm a local run and document gaps**  
   Follow the frontend + training Compose steps above. Record which secrets are missing (`frontend_env`, `DLP/Firebase/Admin_SDK`, AWS SSO, reCAPTCHA, OpenAI) without committing values. Update this README if a command fails for a reason not already listed.

2. **Diagnose [datasciencegt-dlp.com](https://datasciencegt-dlp.com) or choose a new host**  
   The hostname currently returns HTTP 500. Check DNS, the ALB in `next.config.js`, ECS workflows (`.github/workflows/aws.yml` deploys on branch `prod-deploy` to cluster `deep-learning-playground` in `us-west-2`; other workflows use `us-east-1`), and whether DS@GT still owns the domain. If the old AWS account is gone, document a new deploy target (club Vercel/GCP/AWS) instead of pretending the 2024 pipeline still applies.

3. **Dependency, Node, and security audit**  
   `nextjs` is frozen at late-2024 Dependabot merges. Unmerged branches on this remote include bumps such as Next `14.2.21` and Django `4.2.18` (January 2025). Root `Dockerfile` vs empty yarn scripts, Python `<3.10`, SST 2.40, and `CODEOWNERS` pointing at another org are all in scope. Run `pnpm audit` / `poetry` audits locally; do not invent CVE counts in docs.

4. **First-time contributor onboarding**  
   Replace Discord/GitGuardian/AWS “message a Project Lead” steps that assume a staffed DLP Discord. File starter issues (good first issues): empty `dlp-cli` submodule, missing `.github/readme_images/` (old README screenshots), Flask vs Django port collision, `/api/train` and `/api/aws` rewrite gap, stale GitHub URLs in `constants.ts`. Use `.github/ISSUE_TEMPLATE/` (bug, feature, team-wide task).

5. **Upstream and leftover branches**  
   `nextjs` matches upstream `nextjs` today, so there is nothing to pull on that branch until upstream moves. **Hypothesis:** useful leftover work is more likely on this repo’s other branches (`dev`, feature branches, Dependabot) than on DSGT-DLP `nextjs`. Compare before cherry-picking. Keep the fork link either way.

6. **Recruit a project lead and a weekly meeting**  
   No Fall 2026 student lead yet. Coordinate with Aamogh Sawant and Samantha Forero. DS@GT events should be **after 6:30 PM ET**. Until a lead exists, send access questions to [hello@datasciencegt.org](mailto:hello@datasciencegt.org).

7. **Hypothesis: split “what is the backend?”**  
   Flask (`backend/`) and Django (`training/`) both speak `/api` on 8000. Tabular training from the UI posts to `/api/training/tabular` (Django). Image training posts to `/api/train/img-run` (Flask path, no Next rewrite). A member could map each frontend RTK Query URL to one process and delete or proxy the rest.

## How to join and contribute

1. Email [hello@datasciencegt.org](mailto:hello@datasciencegt.org) (or Samantha Forero / Aamogh Sawant) so someone can add you to GitHub org access. `CODEOWNERS` still requests `@DSGT-DLP/project-lead`; that team may not exist on this org.
2. Use this repo: [https://github.com/DataScience-GT/Deep-Learning-Playground](https://github.com/DataScience-GT/Deep-Learning-Playground).
3. Branch from `nextjs`. `create-branch.sh <name>` does stash → checkout `nextjs` → pull → `git checkout -b`.
4. Open an issue first for non-trivial work (templates under `.github/ISSUE_TEMPLATE/`).
5. Send a pull request into `nextjs`. Fill `.github/pull_request_template.md`: issue number, user problem, solution, how you tested.
6. Do not commit secrets. Install pre-commit if you will touch Python deps: `pip install pre-commit ggshield` then `pre-commit install`. `ggshield auth login` needs a GitGuardian account the club may or may not still sponsor.
7. Be kind. See [`.github/CODE_OF_CONDUCT.md`](.github/CODE_OF_CONDUCT.md).

Good first moves: get `pnpm start` running, read `frontend/src/pages/about.tsx` and `frontend/src/pages/train/`, skim `.github/Architecture.md`, then pick a work stream and file an issue.

## License

Application code is MIT; see [LICENSE](./LICENSE). Documentation in this repo is Creative Commons as in [`.github/LICENSE-docs`](.github/LICENSE-docs).
