#!/usr/bin/env bash
set -euo pipefail

############################################################
# Helper: detect package manager (apt, dnf, yum, zypper)
############################################################
if command -v apt >/dev/null 2>&1; then
  PKG_UPDATE='sudo apt update -y'
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
sudo service postgresql start

echo "[INFO] Creating PostgreSQL database & user..."
sudo -u postgres psql <<'EOSQL'
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
EOSQL
echo "[INFO] PostgreSQL ready → try: psql -U holocleanuser -W holo"

############################################################
# 4. MySQL
############################################################
echo "[STEP 4] Installing MySQL Server..."
eval "$PKG_INSTALL mysql-server"

echo "[INFO] Starting MySQL service..."
sudo service mysql start

echo "[INFO] Configuring MySQL root password and sample DB..."
sudo mysql -u root <<'EOFMYSQL'
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '5ZSL45ZS28uvI3^#zv#l';
FLUSH PRIVILEGES;

CREATE DATABASE IF NOT EXISTS mydb;
/* Optional user example:
   CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'mypassword';
   GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'localhost';
   FLUSH PRIVILEGES;
*/
EOFMYSQL
echo "[INFO] MySQL ready → login with: mysql -u root -p"

############################################################
# 5. Miniconda (direct download)
############################################################
echo "[STEP 5] Installing Miniconda..."
cd /root
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p /root/miniconda3
eval "$(/root/miniconda3/bin/conda shell.bash hook)"

############################################################
# 6. Global pip config (direct to PyPI)
############################################################
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'PIPCONF'
[global]
index-url = https://pypi.org/simple
PIPCONF

############################################################
# 7. Optional PYTHONPATH
############################################################
echo 'export PYTHONPATH=/root/AutoMLClustering' >> /root/.bashrc

############################################################
# 8. Create env from environment.yml (if exists)
############################################################
if [[ -f /root/AutoMLClustering/environment.yml ]]; then
  echo "[STEP 8] Creating Conda env from environment.yml..."
  conda env create -f /root/AutoMLClustering/environment.yml
fi

echo "[INFO] Installing raha into (base)..."
pip install raha

echo "[INFO] Installing MySQL Python connector..."
pip install mysql-connector-python

############################################################
# 9. Extra Conda environments
############################################################
echo "[STEP 9] Creating hc37 (Python 3.7)..."
conda create -y -n hc37 python=3.7

echo "[STEP 10] Creating activedetect (Python 2.7)..."
conda create -y -n activedetect python=2.7

############################################################
# 11. Install HoloClean in hc37
############################################################
echo "[INFO] Activating hc37..."
conda activate hc37
cd /root/AutoMLClustering/src/cleaning/holoclean-master
pip install -r requirements.txt

############################################################
# 12. Install BoostClean in activedetect
############################################################
echo "[INFO] Switching to activedetect..."
conda deactivate
conda activate activedetect
cd /root/AutoMLClustering/src/cleaning/BoostClean
pip install -e .

############################################################
# 13. Activate torch110 env (assumes it exists)
############################################################
conda deactivate
echo "[INFO] Activating torch110..."
conda activate torch110

############################################################
# 14. Finish
############################################################
cd /root/AutoMLClustering
echo "[SUCCESS] Installation & configuration complete."
cat <<'EOT'
-----------------------------------------------------------------
  1) PostgreSQL: database 'holo' & user 'holocleanuser' created.
  2) MySQL: database 'mydb' created, root password set.
  3) HoloClean installed in env 'hc37'.
  4) BoostClean installed in env 'activedetect' (Python 2.7).
  5) Current env: 'torch110'.
     Use:
       conda activate hc37
       conda activate activedetect
       conda activate torch110
-----------------------------------------------------------------
EOT

# Initialize Conda for future shells
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
