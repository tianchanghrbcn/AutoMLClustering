#!/bin/bash
set -e  # 遇到错误就退出脚本

#######################################
# 1. 更新 apt 源为清华镜像源
#######################################
echo "[INFO] 正在备份并更新 apt 源为清华源..."
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo bash -c 'cat > /etc/apt/sources.list' << EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
EOF

sudo apt update

#######################################
# 2. 安装必要的系统依赖库
#######################################
echo "[INFO] 安装必要的系统依赖库..."
sudo apt install -y software-properties-common libatlas-base-dev libblas-dev liblapack-dev gfortran curl

#######################################
# 3. 安装 PostgreSQL 并配置数据库
#######################################
echo "[INFO] 安装 PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib

echo "[INFO] 启动 PostgreSQL 服务..."
sudo service postgresql start

echo "[INFO] 创建 PostgreSQL 数据库和用户..."
sudo -u postgres psql <<EOF
CREATE DATABASE holo;
CREATE USER holocleanuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES ON DATABASE holo TO holocleanuser;
\c holo
ALTER SCHEMA public OWNER TO holocleanuser;
EOF

echo "[INFO] PostgreSQL 配置完成，可以使用 psql -U holocleanuser -W holo 连接数据库"

#######################################
# ========== 新增：安装 MySQL 并配置 ==========
#######################################
echo "[INFO] 安装 MySQL Server..."
sudo apt install -y mysql-server

echo "[INFO] 启动 MySQL 服务..."
sudo service mysql start

echo "[INFO] 正在设置 MySQL root 密码并创建 mydb 数据库 (示例)"
# 注意：如果你已经设置过 root 密码，此处可能需要改为 "sudo mysql -u root -p"
sudo mysql -u root <<EOFMYSQL
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '5ZSL45ZS28uvI3^#zv#l';
FLUSH PRIVILEGES;

CREATE DATABASE IF NOT EXISTS mydb;
-- 如果你想创建其他用户，也可以:
-- CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'mypassword';
-- GRANT ALL PRIVILEGES ON mydb.* TO 'myuser'@'localhost';
-- FLUSH PRIVILEGES;
EOFMYSQL

echo "[INFO] MySQL 安装并初步配置完成！可以使用命令登陆："
echo "      mysql -u root -pMyRootPassword123"

#######################################
# 4. 在 /root/ 目录安装 Miniconda3
#######################################
echo "[INFO] 正在下载并安装 Miniconda3..."
cd /root
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p /root/miniconda3

#######################################
# 5. 初始化 conda（推荐用 eval 方式）
#######################################
echo "[INFO] 初始化 conda (eval 方法)..."
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
# 上面这句会在当前 Shell 中加载 conda，保证后续 `conda activate` 可用

#######################################
# 6. 设置 PYTHONPATH（根据需要）
#######################################
echo 'export PYTHONPATH=/root/AutoMLClustering' >> /root/.bashrc

#######################################
# 7. 使用 environment.yml 创建环境
#######################################
if [ -f "/root/AutoMLClustering/environment.yml" ]; then
    echo "[INFO] 检测到 environment.yml，正在创建 Conda 环境..."
    conda env create -f /root/AutoMLClustering/environment.yml
fi

echo "[INFO] 在当前 (base) 环境安装 raha..."
pip install raha -i https://pypi.tuna.tsinghua.edu.cn/simple

#######################################
# ========== 新增：安装 MySQL Python 连接库 ==========
#######################################
echo "[INFO] 安装 mysql-connector-python 以供 Python 访问 MySQL..."
pip install mysql-connector-python

#######################################
# 8. 创建 hc37 环境 (Python 3.7)
#######################################
echo "[INFO] 创建 hc37 (Python 3.7) 环境..."
conda create -y -n hc37 python=3.7

#######################################
# 9. 创建 activedetect 环境 (Python 2.7)
#######################################
echo "[INFO] 创建 activedetect (Python 2.7) 环境..."
conda create -y -n activedetect python=2.7

#######################################
# 10. 进入 hc37 环境并安装 HoloClean
#######################################
echo "[INFO] 激活 hc37 环境..."
conda activate hc37

echo "[INFO] 进入 HoloClean 目录并安装依赖..."
cd /root/AutoMLClustering/src/cleaning/holoclean-master
pip install -r requirements.txt

#######################################
# 11. 激活 activedetect 环境并安装 BoostClean
#######################################
echo "[INFO] 切换到 activedetect (Python 2.7) 环境..."
conda deactivate
conda activate activedetect

echo "[INFO] 进入 BoostClean 目录并运行 setup.py..."
cd /root/AutoMLClustering/src/cleaning/BoostClean
pip install -e .

#######################################
# 12. 切换到 torch110 环境
#######################################
conda deactivate
echo "[INFO] 激活 torch110 环境..."
conda activate torch110

#######################################
# 13. 回到 /root/AutoMLClustering 并提示完成
#######################################
cd /root/AutoMLClustering
echo "[INFO] 安装和配置完成！"
echo "-----------------------------------------------------"
echo "   1) PostgreSQL 已安装并配置数据库 holo/holocleanuser."
echo "   2) MySQL 已安装并创建 mydb 数据库，root 密码: MyRootPassword123"
echo "   3) HoloClean 已安装到 hc37 环境."
echo "   4) activedetect (Python2.7) 环境下已安装 BoostClean."
echo "   当前环境: torch110."
echo "   你可以使用以下命令手动切换环境:"
echo "     conda activate hc37"
echo "     conda activate activedetect"
echo "     conda activate torch110"
echo "-----------------------------------------------------"
