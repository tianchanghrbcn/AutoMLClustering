import os
import math
import time
import numpy as np
import pandas as pd
import optuna

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    AffinityPropagation,
    DBSCAN,
    OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.metrics.pairwise import cosine_distances


# ========== 1) 辅助函数：解析 k 的特殊范围字符串 ==========
def get_cluster_range(n: int, k_str: str):
    """
    根据给定的 k_str（例如 '> sqrt(n)', '(sqrt(n)/2, sqrt(n)]', '≤ sqrt(n)/2'）
    返回 (n_min, n_max) 作为聚类簇数的搜索区间，满足：
      - 当 k_str == '> sqrt(n)' 时，搜索范围为 [ceil(sqrt(n)), 2*ceil(sqrt(n))]
      - 当 k_str == '(sqrt(n)/2, sqrt(n)]' 时，范围为 [floor(sqrt(n)/2)+1, ceil(sqrt(n))]
      - 当 k_str == '≤ sqrt(n)/2' 时，范围为 [2, floor(sqrt(n)/2)]
      - 其他情况返回默认 [2, min(n, 50)]
    """
    sqrt_n = math.sqrt(n)
    if k_str == "> sqrt(n)":
        lower = int(math.ceil(sqrt_n))
        upper = int(2 * math.ceil(sqrt_n))  # 2倍根号n（n通常为几千，不会越界）
        return lower, upper
    elif k_str == "(sqrt(n)/2, sqrt(n)]":
        lower = int(math.floor(sqrt_n / 2)) + 1
        upper = int(math.ceil(sqrt_n))
        if lower > upper:
            upper = lower
        return lower, upper
    elif k_str == "≤ sqrt(n)/2":
        lower = 2
        upper = int(math.floor(sqrt_n / 2))
        if lower > upper:
            upper = lower
        return lower, upper
    else:
        return 2, min(n, 50)


# ========== 2) 辅助函数：通用字符串区间解析 ==========
def parse_range(value_str: str, is_int: bool = False):
    """
    将类似 '0.7-0.9'、'0.7 to 0.9'、'5-25'、'5 to 25' 等字符串解析成 (low, high)。
    is_int=True 表示解析为整数区间，否则解析为浮点区间。
    若解析失败则抛出 ValueError。
    """
    if ' to ' in value_str:
        parts = value_str.split(' to ')
    elif '-' in value_str:
        parts = value_str.split('-')
    else:
        raise ValueError(f"无法解析区间参数: {value_str}")
    if len(parts) != 2:
        raise ValueError(f"解析失败，找不到正确的上下界: {value_str}")
    low_str, high_str = parts[0].strip(), parts[1].strip()
    if is_int:
        return int(low_str), int(high_str)
    else:
        return float(low_str), float(high_str)


# ========== 3) 数据预处理 ==========
def preprocess_data(cleaned_file_path: str):
    """
    预处理数据：
      1. 读取 CSV 文件，并检查是否存在；
      2. 排除列名中包含 'id' 的列；
      3. 对类别型特征做频率编码；
      4. 删除缺失数据；
      5. 标准化；
    返回：X_scaled (np.array), 数据行数 n
    """
    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"[ERROR] 清洗后的文件不存在: {cleaned_file_path}")
    df = pd.read_csv(cleaned_file_path)
    excluded_columns = [col for col in df.columns if 'id' in col.lower()]
    remaining_columns = df.columns.difference(excluded_columns)
    X = df[remaining_columns].copy()
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype).startswith('category'):
            freq_map = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq_map).fillna(0)
    X.dropna(inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, len(X)


# ========== 4) 保存结果 ==========
def save_results(cleaned_file_path: str, dataset_id: str, algorithm: str, params: dict,
                 final_labels: np.ndarray, final_db_score: float, final_silhouette_score: float,
                 alpha: float, beta: float, start_time: float):
    """
    计算最终综合得分并保存结果到文本文件，返回 (输出目录, 运行时长)。
    """
    final_db_score = max(final_db_score, 1e-12)
    combined_score = alpha * (1 / final_db_score) + beta * final_silhouette_score
    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    base_filename = os.path.splitext(os.path.basename(cleaned_file_path))[0]
    output_dir = os.path.join(os.getcwd(), "..", "..", "..",
                              "results", "test_clustered_data",
                              algorithm.upper(), f"clustered_{dataset_id}")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f"{base_filename}.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"Best parameters: {params_str}\n")
        f.write(f"Number of clusters: {len(set(final_labels))}\n")
        f.write(f"Final Combined Score: {combined_score}\n")
        f.write(f"Final Silhouette Score: {final_silhouette_score}\n")
        f.write(f"Final Davies-Bouldin Score: {final_db_score}\n")
    print(f"[INFO] 结果已保存到: {result_file}")
    run_time = time.time() - start_time
    return output_dir, run_time


# ========== 5) 主函数：自动聚类与调参 ==========
def run_clustering_test(dataset_id: str, algorithm: str, params: dict,
                        cleaned_file_path: str, alpha: float = 0.75, beta: float = 0.25):
    """
    根据算法类型进行 optuna 参数调优搜索，搜索范围均使用全范围：
      - 对于簇数参数：范围为 (2, min(n,50))
      - 对于其他参数，若为数值参数，则采用预设全范围（例如 AP: damping (0.5,1.0), preference (-1000,0)；
        DBSCAN: eps (0.1,10), min_samples (2,50);
        OPTICS: min_samples (2,50), xi (0.01,0.5), min_cluster_size (0.01,0.5)）。
    调优得到的最优参数如果不在预测范围内（预测范围由传入参数决定），则在保存结果时记录 note 信息。
    """
    start_time = time.time()
    algo = algorithm.upper()
    try:
        X_scaled, n = preprocess_data(cleaned_file_path)

        # 辅助函数，判断是否不在预测范围内
        def record_outside(predicted_range, best_val):
            return not (predicted_range[0] <= best_val <= predicted_range[1])

        # 对于簇数相关的算法，全范围搜索范围为 (2, min(n, 50))
        full_range = (2, min(n, 50))

        # ------------------- KMEANS -------------------
        if algo == "KMEANS":
            k_str = params.get("k", None)
            # 预测范围仅作参考，如果有特殊预测字符串
            if k_str is not None and k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                predicted_range = get_cluster_range(n, k_str)
            elif k_str is not None:
                predicted_range = parse_range(k_str, is_int=True)
            else:
                predicted_range = full_range
            print(f"[INFO] KMEANS全范围搜索区间: {full_range}, 预测范围: {predicted_range}")

            def objective(trial):
                k_ = trial.suggest_int("n_clusters", full_range[0], full_range[1])
                labels_ = KMeans(n_clusters=k_, init='k-means++', n_init=10, random_state=0).fit_predict(X_scaled)
                if len(set(labels_)) <= 1:
                    return -float('inf')
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_k = study.best_params["n_clusters"]
            note = ""
            if record_outside(predicted_range, best_k):
                note = f"预测范围为{predicted_range}，但最优参数为{best_k}"
            final_labels = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=0).fit_predict(X_scaled)
            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)
            params_out = {"k": k_str if k_str else f"{predicted_range}",
                          "best_n_clusters": best_k}
            if note:
                params_out["note"] = note
            return save_results(cleaned_file_path, dataset_id, "KMEANS",
                                params_out,
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)

        # ------------------- GMM -------------------
        elif algo == "GMM":
            k_str = params.get("k", None)
            if k_str is not None and k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                predicted_range = get_cluster_range(n, k_str)
            elif k_str is not None:
                predicted_range = parse_range(k_str, is_int=True)
            else:
                predicted_range = full_range
            print(f"[INFO] GMM全范围搜索区间: {full_range}, 预测范围: {predicted_range}")
            covariance_type = params.get("covariance_type", "full")

            def objective(trial):
                k_ = trial.suggest_int("n_components", full_range[0], full_range[1])
                labels_ = GaussianMixture(n_components=k_, covariance_type=covariance_type, random_state=0) \
                    .fit_predict(X_scaled)
                if len(np.unique(labels_)) <= 1:
                    return -float('inf')
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_n_components = study.best_params["n_components"]
            note = ""
            if record_outside(predicted_range, best_n_components):
                note = f"预测范围为{predicted_range}，但最优参数为{best_n_components}"
            final_labels = GaussianMixture(n_components=best_n_components, covariance_type=covariance_type,
                                           random_state=0) \
                .fit_predict(X_scaled)
            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)
            params_out = {"k": k_str if k_str else f"{predicted_range}",
                          "best_n_components": best_n_components,
                          "covariance_type": covariance_type}
            if note:
                params_out["note"] = note
            return save_results(cleaned_file_path, dataset_id, "GMM",
                                params_out,
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)

        # ------------------- HC (层次聚类) -------------------
        elif algo == "HC":
            k_str = params.get("k", None)
            if k_str is not None and k_str in {"> sqrt(n)", "(sqrt(n)/2, sqrt(n)]", "≤ sqrt(n)/2"}:
                predicted_range = get_cluster_range(n, k_str)
            elif k_str is not None:
                predicted_range = parse_range(k_str, is_int=True)
            else:
                predicted_range = full_range
            print(f"[INFO] HC全范围搜索区间: {full_range}, 预测范围: {predicted_range}")
            linkage = params.get("linkage", "ward")
            metric = params.get("metric", "euclidean")
            if linkage == "ward" and metric != "euclidean":
                raise ValueError("[ERROR] Ward linkage only supports 'euclidean' metric.")

            def objective(trial):
                k_ = trial.suggest_int("n_clusters", full_range[0], full_range[1])
                labels_ = AgglomerativeClustering(n_clusters=k_, linkage=linkage, metric=metric) \
                    .fit_predict(X_scaled)
                if len(np.unique(labels_)) <= 1:
                    return -float('inf')
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_k = study.best_params["n_clusters"]
            note = ""
            if record_outside(predicted_range, best_k):
                note = f"预测范围为{predicted_range}，但最优参数为{best_k}"
            final_labels = AgglomerativeClustering(n_clusters=best_k, linkage=linkage, metric=metric) \
                .fit_predict(X_scaled)
            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)
            params_out = {"k": k_str if k_str else f"{predicted_range}",
                          "best_n_clusters": best_k,
                          "linkage": linkage,
                          "metric": metric}
            if note:
                params_out["note"] = note
            return save_results(cleaned_file_path, dataset_id, "HC",
                                params_out,
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)

        # ------------------- AP (AffinityPropagation) -------------------
        elif algo == "AP":
            # 全范围搜索，给出较宽的默认范围：
            damping_range = (0.5, 1.0)
            preference_range = (-1000, 0)
            n_trials = params.get("n_trials", 50)
            print(f"[INFO] AP搜索范围: damping {damping_range}, preference {preference_range}")

            def objective(trial):
                damping_ = trial.suggest_float("damping", damping_range[0], damping_range[1])
                preference_ = trial.suggest_int("preference", preference_range[0], preference_range[1])
                ap_ = AffinityPropagation(damping=damping_, preference=preference_, random_state=0)
                labels_ = ap_.fit_predict(X_scaled)
                n_clusters_ = len(np.unique(labels_))
                if n_clusters_ <= 1 or n_clusters_ >= len(X_scaled):
                    return float('-inf')
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return alpha * (1 / max(db_, 1e-12)) + beta * sil_

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params_ = study.best_params
            best_damping = best_params_["damping"]
            best_preference = best_params_["preference"]
            best_ap = AffinityPropagation(damping=best_damping, preference=best_preference, random_state=0)
            final_labels = best_ap.fit_predict(X_scaled)
            final_db_score = davies_bouldin_score(X_scaled, final_labels)
            final_silhouette_score = silhouette_score(X_scaled, final_labels)
            return save_results(cleaned_file_path, dataset_id, "AP",
                                {"damping": damping_range, "preference": preference_range,
                                 "best_damping": best_damping, "best_preference": best_preference},
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)

        # ------------------- DBSCAN -------------------
        elif algo == "DBSCAN":
            # 全范围搜索： eps (0.1, 10.0)，min_samples (2, 50)
            eps_range = (0.1, 10.0)
            ms_range = (2, 50)
            n_trials = params.get("n_trials", 50)
            print(f"[INFO] DBSCAN搜索范围: eps {eps_range}, min_samples {ms_range}")

            def objective(trial):
                eps_ = trial.suggest_float("eps", eps_range[0], eps_range[1])
                min_s_ = trial.suggest_int("min_samples", ms_range[0], ms_range[1])
                dbscan_ = DBSCAN(eps=eps_, min_samples=min_s_, metric='euclidean')
                labels_ = dbscan_.fit_predict(X_scaled)
                n_clusters_ = len(np.unique(labels_)) - (1 if -1 in labels_ else 0)
                if n_clusters_ < 2:
                    return float('-inf')
                noise_ratio = np.mean(labels_ == -1)
                noise_penalty = max(0, 1 - noise_ratio)
                db_ = davies_bouldin_score(X_scaled, labels_)
                sil_ = silhouette_score(X_scaled, labels_)
                return (alpha * (1 / max(db_, 1e-12)) + beta * sil_) * noise_penalty

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params_ = study.best_params
            best_eps = best_params_["eps"]
            best_min_samples = best_params_["min_samples"]
            best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean')
            final_labels = best_dbscan.fit_predict(X_scaled)
            n_clusters_final = len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)
            if n_clusters_final > 1:
                final_db_score = davies_bouldin_score(X_scaled, final_labels)
                final_silhouette_score = silhouette_score(X_scaled, final_labels)
            else:
                final_db_score = float('inf')
                final_silhouette_score = 0.0
            return save_results(cleaned_file_path, dataset_id, "DBSCAN",
                                {"eps": eps_range, "min_samples": ms_range,
                                 "best_eps": best_eps, "best_min_samples": best_min_samples},
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)

        # ------------------- OPTICS -------------------
        elif algo == "OPTICS":
            # 全范围搜索： min_samples (2,50), xi (0.01,0.5), min_cluster_size (0.01,0.5)
            ms_range = (2, 50)
            xi_range = (0.01, 0.5)
            mcs_range = (0.01, 0.5)
            n_trials = params.get("n_trials", 50)
            print(f"[INFO] OPTICS搜索范围: min_samples {ms_range}, xi {xi_range}, min_cluster_size {mcs_range}")
            X_cosine = cosine_distances(X_scaled)

            def objective(trial):
                ms_ = trial.suggest_int("min_samples", ms_range[0], ms_range[1])
                xi_ = trial.suggest_float("xi", xi_range[0], xi_range[1])
                mcs_ = trial.suggest_float("min_cluster_size", mcs_range[0], mcs_range[1])
                optics_ = OPTICS(min_samples=ms_, xi=xi_, min_cluster_size=mcs_, metric='precomputed')
                optics_.fit(X_cosine)
                labels_ = optics_.labels_
                n_clusters_ = len(np.unique(labels_)) - (1 if -1 in labels_ else 0)
                if n_clusters_ < 2:
                    return float('-inf')
                noise_ratio = np.mean(labels_ == -1)
                noise_penalty = max(0, 1 - noise_ratio)
                sil_ = silhouette_score(X_cosine, labels_, metric='precomputed')
                db_ = davies_bouldin_score(X_scaled, labels_)
                return (alpha * (1 / max(db_, 1e-12)) + beta * sil_) * noise_penalty

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            best_params_ = study.best_params
            best_ms = best_params_["min_samples"]
            best_xi = best_params_["xi"]
            best_mcs = best_params_["min_cluster_size"]
            final_optics = OPTICS(min_samples=best_ms, xi=best_xi, min_cluster_size=best_mcs, metric='precomputed')
            final_optics.fit(X_cosine)
            final_labels = final_optics.labels_
            n_clusters_final = len(np.unique(final_labels)) - (1 if -1 in final_labels else 0)
            if n_clusters_final > 1:
                final_silhouette_score = silhouette_score(X_cosine, final_labels, metric='precomputed')
                final_db_score = davies_bouldin_score(X_scaled, final_labels)
            else:
                final_silhouette_score = 0.0
                final_db_score = float('inf')
            return save_results(cleaned_file_path, dataset_id, "OPTICS",
                                {"min_samples": ms_range, "xi": xi_range, "min_cluster_size": mcs_range,
                                 "best_min_samples": best_ms, "best_xi": best_xi, "best_min_cluster_size": best_mcs},
                                final_labels, final_db_score, final_silhouette_score,
                                alpha, beta, start_time)
        else:
            raise ValueError(f"[ERROR] 不支持的算法类型: {algorithm}")
    except Exception as e:
        print(f"[ERROR] 聚类时出错: {e}")
        return None, None
