#!/usr/bin/env bash
set -euo pipefail

############################################################
# 0. Paths / variables (NO hard-coded /root or /home/changtian)
############################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"     # 默认：脚本所在目录就是项目根目录
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"      # 未来你的 miniconda3 就在 $HOME/miniconda3
CONDA_BIN="$CONDA_DIR/bin/conda"

PG_DB="${PG_DB:-holo}"
PG_USER="${PG_USER:-holocleanuser}"
PG_PASSWORD="${PG_PASSWORD:-abcd1234}"

MYSQL_DB="${MYSQL_DB:-mydb}"
MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-5ZSL45ZS28uvI3^#zv#l}"

# small helpers
start_service() {
  local svc="$1"
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable "$svc" >/dev/null 2>&1 || true
    sudo systemctl start "$svc"  >/dev/null 2>&1 || true
  else
    sudo service "$svc" start >/dev/null 2>&1 || true
  fi
}

ensure_line_in_file() {
  local line="$1"
  local file="$2"
  touch "$file"
  grep -qxF "$line" "$file" || echo "$line" >> "$file"
}

conda_env_exists() {
  # prints nothing; return code indicates existence
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$1"
}

mysql_exec() {
  # idempotent: first try passwordless (fresh install / auth_socket), else use password
  if sudo mysql -u root -e "SELECT 1;" >/dev/null 2>&1; then
    sudo mysql -u root
  else
    sudo mysql -u root -p"$MYSQL_ROOT_PASSWORD"
  fi
}

############################################################
# Helper: detect package manager (apt, dnf, yum, zypper)
############################################################
if command -v apt >/dev/null 2>&1; then
  PKG_UPDATE='sudo apt update'
  PKG_INSTALL='sudo apt install -y'
elif command -v dnf >/dev/null 2>&1; then
  PKG_UPDATE='sudo dnf makecache'
  PKG_INSTALL='sudo dnf install -y'
elif command -v yum >/dev/null 2>&1; then
  PKG_UPDATE='sudo yum makecache'
  PKG_INSTALL='sudo yum install -y'
elif command -v zypper >/dev/null 2>&1; then
  PKG_UPDATE='sudo zypper refresh'
  PKG_INSTALL='sudo zypper install -y'
else
  echo "[ERROR] Unsupported Linux distribution. Install dependencies manually."
  exit 1
fi

echo "[STEP 1] Refreshing package index..."
eval "$PKG_UPDATE"

############################################################
# 2. System build tools & libs
############################################################
echo "[STEP 2] Installing base development libraries..."
eval "$PKG_INSTALL software-properties-common libatlas-base-dev libblas-dev liblapack-dev gfortran curl"

############################################################
# 3. PostgreSQL
############################################################
echo "[STEP 3] Installing PostgreSQL..."
eval "$PKG_INSTALL postgresql postgresql-contrib"

echo "[INFO] Starting PostgreSQL service..."
start_service postgresql

echo "[INFO] Creating PostgreSQL database & user (idempotent)..."
sudo -u postgres psql -v ON_ERROR_STOP=1 <<EOSQL
-- create user if not exists
SELECT 'CREATE USER $PG_USER' ||
       ' WITH PASSWORD ''' || '$PG_PASSWORD' || ''';'
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$PG_USER') \gexec

-- ensure password is what we expect (safe to rerun)
ALTER USER $PG_USER WITH PASSWORD '$PG_PASSWORD';

-- create db if not exists, set owner
SELECT 'CREATE DATABASE $PG_DB OWNER $PG_USER;'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '$PG_DB') \gexec

-- ensure ownership (safe to rerun)
ALTER DATABASE $PG_DB OWNER TO $PG_USER;

GRANT ALL PRIVILEGES ON DATABASE $PG_DB TO $PG_USER;
\c $PG_DB
ALTER SCHEMA public OWNER TO $PG_USER;
EOSQL
echo "[INFO] PostgreSQL ready → try: psql -U $PG_USER -W $PG_DB"

############################################################
# 4. MySQL
############################################################
echo "[STEP 4] Installing MySQL Server..."
eval "$PKG_INSTALL mysql-server"

echo "[INFO] Starting MySQL service..."
start_service mysql

echo "[INFO] Configuring MySQL root password and sample DB (idempotent)..."
mysql_exec <<EOFMYSQL
-- Set root auth to mysql_native_password & password (safe to rerun if you can login)
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '${MYSQL_ROOT_PASSWORD}';
FLUSH PRIVILEGES;

CREATE DATABASE IF NOT EXISTS ${MYSQL_DB};
EOFMYSQL
echo "[INFO] MySQL ready → login with: mysql -u root -p"

############################################################
# 5. Miniconda (direct download)
############################################################
echo "[STEP 5] Installing Miniconda (idempotent)..."
mkdir -p "$HOME/.cache/miniconda"
cd "$HOME/.cache/miniconda"

if [[ ! -x "$CONDA_BIN" ]]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b -p "$CONDA_DIR"
else
  echo "[INFO] Miniconda already exists at: $CONDA_DIR (skip install)"
fi

# enable conda in THIS script session
eval "$("$CONDA_BIN" shell.bash hook)"

# ----------------------------------------------------------
# NEW: Auto-accept Anaconda channel Terms of Service (ToS)
#      so the script won't block on prompts.
# ----------------------------------------------------------
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
# If supported by your conda, persist the setting (idempotent).
"$CONDA_BIN" config --set plugins.auto_accept_tos yes >/dev/null 2>&1 || true

############################################################
# 6. Global pip config (direct to PyPI)
############################################################
mkdir -p "$HOME/.pip"
if [[ ! -f "$HOME/.pip/pip.conf" ]]; then
  cat > "$HOME/.pip/pip.conf" <<'PIPCONF'
[global]
index-url = https://pypi.org/simple
PIPCONF
else
  echo "[INFO] pip.conf already exists (skip overwrite): $HOME/.pip/pip.conf"
fi

############################################################
# 7. Optional PYTHONPATH
############################################################
# do NOT write to /root/.bashrc; always $HOME
ensure_line_in_file "export PYTHONPATH=\"$PROJECT_ROOT\"" "$HOME/.bashrc"

############################################################
# 8. Create env from environment.yml (if exists)
############################################################
TORCH_ENV="${TORCH_ENV:-torch110}"

if [[ -f "$PROJECT_ROOT/environment.yml" ]]; then
  echo "[STEP 8] Creating/Updating Conda env from environment.yml (idempotent)..."

  # --------------------------------------------------------
  # NEW: Make tmp file end with .yml so conda's env-spec
  #      plugin can detect it correctly (fixes
  #      EnvironmentSpecPluginNotDetected).
  # --------------------------------------------------------
  ENV_YML_TMP="$(mktemp --suffix=.yml 2>/dev/null || mktemp /tmp/environment.XXXXXX.yml)"

  # remove any 'prefix:' line to avoid absolute-path env binding
  grep -vE '^\s*prefix:\s*' "$PROJECT_ROOT/environment.yml" > "$ENV_YML_TMP"

  # --------------------------------------------------------
  # NEW: Some conda versions require explicitly selecting
  #      the environment spec plugin:
  #        --environment-spec environment.yml   (newer)
  #        --env-spec environment.yml           (some builds)
  #      We auto-detect which flag exists.
  # --------------------------------------------------------
  ENV_SPEC_ARGS=()
  if "$CONDA_BIN" env create --help 2>/dev/null | grep -q -- '--environment-spec'; then
    ENV_SPEC_ARGS+=(--environment-spec environment.yml)
  elif "$CONDA_BIN" env create --help 2>/dev/null | grep -q -- '--env-spec'; then
    ENV_SPEC_ARGS+=(--env-spec environment.yml)
  fi

  # pick env name from yaml if present
  yaml_name="$(awk -F': *' '/^name:/{print $2; exit}' "$ENV_YML_TMP" || true)"
  if [[ -n "${yaml_name:-}" ]]; then
    TORCH_ENV="$yaml_name"
  fi

  if conda_env_exists "$TORCH_ENV"; then
    conda env update -n "$TORCH_ENV" -f "$ENV_YML_TMP" --prune "${ENV_SPEC_ARGS[@]}"
  else
    conda env create -n "$TORCH_ENV" -f "$ENV_YML_TMP" "${ENV_SPEC_ARGS[@]}"
  fi

  rm -f "$ENV_YML_TMP"
else
  echo "[INFO] environment.yml not found at $PROJECT_ROOT/environment.yml (skip STEP 8)"
fi

echo "[INFO] Installing raha into (base)..."
conda run -n base python -m pip install -U raha

echo "[INFO] Installing MySQL Python connector..."
conda run -n base python -m pip install -U mysql-connector-python

############################################################
# 9. Extra Conda environments
############################################################
echo "[STEP 9] Creating hc37 (Python 3.7) (idempotent)..."
if conda_env_exists "hc37"; then
  echo "[INFO] hc37 already exists (skip)"
else
  conda create -y -n hc37 python=3.7
fi

echo "[STEP 10] Creating activedetect (Python 2.7) (idempotent)..."
if conda_env_exists "activedetect"; then
  echo "[INFO] activedetect already exists (skip)"
else
  # python2.7 在新环境可能不可用：先按原结构尝试；失败时给出明确提示
  if ! conda create -y -n activedetect python=2.7; then
    echo "[WARN] Failed to create activedetect (python=2.7)."
    echo "       You may need conda-forge or legacy channel, e.g.:"
    echo "       conda create -y -n activedetect python=2.7 -c conda-forge"
    exit 1
  fi
fi

############################################################
# 11. Install HoloClean in hc37
############################################################
echo "[INFO] Activating hc37..."
conda activate hc37
cd "$PROJECT_ROOT/src/cleaning/holoclean-master"
python -m pip install -r requirements.txt

############################################################
# 12. Install BoostClean in activedetect
############################################################
echo "[INFO] Switching to activedetect..."
conda deactivate
conda activate activedetect
cd "$PROJECT_ROOT/src/cleaning/BoostClean"
python -m pip install -e .

############################################################
# 13. Activate torch env (assumes it exists / created in STEP 8)
############################################################
conda deactivate
echo "[INFO] Activating ${TORCH_ENV}..."
if conda_env_exists "$TORCH_ENV"; then
  conda activate "$TORCH_ENV"
else
  echo "[ERROR] Env '${TORCH_ENV}' does not exist. Check your environment.yml / STEP 8."
  exit 1
fi

#######################################
# 14. Return to project root and display completion message
#######################################
cd "$PROJECT_ROOT"
echo "[INFO] Installation and configuration completed!"
echo "-----------------------------------------------------"
echo "   1) PostgreSQL installed and database ${PG_DB}/${PG_USER} configured."
echo "   2) MySQL installed and database ${MYSQL_DB} created (root password set in script variable)."
echo "   3) HoloClean installed in the hc37 environment."
echo "   4) BoostClean installed in the activedetect (Python 2.7) environment."
echo "   Current environment (in this script session): ${TORCH_ENV}."
echo "   You can manually switch environments using the following commands:"
echo "     conda activate hc37"
echo "     conda activate activedetect"
echo "     conda activate ${TORCH_ENV}"
echo "-----------------------------------------------------"

# conda init is safe to rerun; DO NOT exec a new shell (keeps script rerunnable/non-interactive safe)
"$CONDA_BIN" init bash >/dev/null 2>&1 || true
echo "[INFO] If you want conda to work in new terminals, run:"
echo "       source \"$HOME/.bashrc\""
