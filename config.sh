#!/bin/bash
set -e  # 遇到错误就退出脚本

#######################################
# 1. 初始化 conda（推荐用 eval 方式）
#######################################
echo "[INFO] 初始化 conda (eval 方法)..."
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
# 上面这句会在当前 Shell 中加载 conda，保证后续 `conda activate` 可用

#######################################
# 2. 设置 PYTHONPATH（根据需要）
#######################################
echo 'export PYTHONPATH=/root/AutoMLClustering' >> /root/.bashrc

#######################################
# 3. 使用 environment.yml 创建环境
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
# 4. 创建 hc37 环境 (Python 3.7)
#######################################
echo "[INFO] 创建 hc37 (Python 3.7) 环境..."
conda create -y -n hc37 python=3.7

#######################################
# 5. 创建 activedetect 环境 (Python 2.7)
#######################################
echo "[INFO] 创建 activedetect (Python 2.7) 环境..."
conda create -y -n activedetect python=2.7

#######################################
# 6. 进入 hc37 环境并安装 HoloClean
#######################################
echo "[INFO] 激活 hc37 环境..."
conda activate hc37

echo "[INFO] 进入 HoloClean 目录并安装依赖..."
cd /root/AutoMLClustering/src/cleaning/holoclean-master
pip install -r requirements.txt

#######################################
# 7. 激活 activedetect 环境并安装 BoostClean
#######################################
echo "[INFO] 切换到 activedetect (Python 2.7) 环境..."
conda deactivate
conda activate activedetect

echo "[INFO] 进入 BoostClean 目录并运行 setup.py..."
cd /root/AutoMLClustering/src/cleaning/BoostClean
pip install -e .

#######################################
# 8. 切换到 torch110 环境
#######################################
conda deactivate
echo "[INFO] 激活 torch110 环境..."
conda activate torch110

#######################################
# 9. 回到 /root/AutoMLClustering 并提示完成
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
$HOME/miniconda3/bin/conda init bash
exec "$SHELL"
conda activate torch110
