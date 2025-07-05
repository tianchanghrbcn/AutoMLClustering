import os
import subprocess
from enum import Enum
import time
import re  # 用于解析输出

class ClusterMethod(Enum):
    KMEANSNF = 4         # Affinity Propagation
    DBSCAN = 1     # Density-Based Spatial Clustering
    GMM = 2        # Gaussian Mixture Model
    HC = 0         # Hierarchical Clustering
    KMEANS = 3     # K-means Clustering
    KMEANSPPS = 5     # Ordering Points To Identify Cluster Structure

def run_clustering(dataset_id, algorithm, cluster_method_id, cleaned_file_path):

    try:
        # 根据 cluster_method_id 获取聚类方法名称
        cluster_method = ClusterMethod(cluster_method_id).name
        cluster_script_path = os.path.join("..", "..", "clustering", cluster_method, f"{cluster_method}.py")

        if not os.path.exists(cluster_script_path):
            print(f"[ERROR] 聚类算法脚本未找到: {cluster_script_path}")
            return None, None

        # 设置环境变量
        os.environ["CSV_FILE_PATH"] = cleaned_file_path
        os.environ["DATASET_ID"] = str(dataset_id)
        os.environ["ALGO"] = algorithm

        # 设置结果输出目录
        output_dir = os.path.join(os.getcwd(), "..", "..", "..", "results", "clustered_data", cluster_method, algorithm,
                                  f"clustered_{dataset_id}")
        os.makedirs(output_dir, exist_ok=True)
        os.environ["OUTPUT_DIR"] = output_dir

        # 设置命令
        command = ["python", cluster_script_path]

        print(f"[INFO] 运行聚类算法: {cluster_method}，数据集编号: {dataset_id}, 清洗算法: {algorithm}")

        # 开始计时
        start_time = time.time()

        # 执行聚类脚本
        result = subprocess.run(command, capture_output=True, text=True, check=True, env=os.environ)

        # 提取脚本中的运行时间
        stdout = result.stdout
        match = re.search(r"Program completed in: ([\d.]+) seconds", stdout)
        if match:
            runtime = float(match.group(1))  # 使用脚本输出的时间
        else:
            # 如果没有找到时间信息，回退到计算的时间
            runtime = time.time() - start_time

        print(f"[INFO] 聚类算法输出:\n{stdout}")

        return output_dir, runtime

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 聚类算法运行错误: {e.stderr}")
        return None, None
    except ValueError as ve:
        print(f"[ERROR] 无效的聚类方法 ID: {cluster_method_id} - {ve}")
        return None, None
    except Exception as ex:
        print(f"[ERROR] 聚类算法执行过程中发生错误: {ex}")
        return None, None
