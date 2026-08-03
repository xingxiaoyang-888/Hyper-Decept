# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import ast
import csv
import json
import argparse
import asyncio
import logging
import os
import random
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict
from collections import Counter

import numpy as np
import pandas as pd
from colorama import Back
from yaml import safe_load
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

from camel.messages import BaseMessage
from oasis.clock.clock import Clock
from oasis.inference.inference_manager import InferencerManager
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType
from oasis.social_platform.task_blackboard import TaskBlackboard
from oasis.social_platform.post_stats import SharedMemory, TweetStats, PostStats
from oasis.social_platform.activation_scheduler import (
    BudgetedActivationScheduler,
    pending_bad_member_ids,
)
from utils.tweet_stats_visualization import visualize_tweet_stats
from task_audit_logger import TaskAuditLogger, patch_agents_with_task_logger

import logging
import os
from datetime import datetime
import pickle

# ====================== 独立评价指标日志 ======================
os.makedirs("log", exist_ok=True)
# 日志会保存在项目根目录：evaluation_metrics.log
metrics_logger = logging.getLogger("evaluation_metrics")
metrics_logger.setLevel(logging.INFO)
metrics_logger.handlers.clear()

# 文件处理器
log_file_path = os.path.join(os.getcwd(), "evaluation_metrics.log")
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
metrics_logger.addHandler(file_handler)

#del os.environ["http_proxy"]
#del os.environ["HTTP_PROXY"]
# del os.environ["https_proxy"]
# del os.environ["HTTPS_PROXY"]

social_log = logging.getLogger(name="social")
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler(
    f"./log/social-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
)
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")
WARNING_MESSAGE = "[Important] Warning: This post is controversial and may provoke debate. Please read critically and verify information independently."
COLLAPSE_POST_MESSAGE = "This post has been collapsed due to the spread of false information, which constitutes a serious violation of the social media platform’s rules. The platform advises users not to like, share, or comment on this post. The original content of the post is as follows: "

def generate_embeddings(texts, model_path="all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    
    embeddings = model.encode(texts)
    
    return embeddings

def kmeans_clustering(embeddings, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans

def save_embeddings_and_clusters(embeddings, clusters, filename='embeddings_clusters_data.pkl'):
    """
    保存嵌入向量和聚类结果到文件
    
    参数:
    embeddings: 嵌入向量数组
    clusters: 聚类标签数组
    filename: 保存的文件名
    """
    data = {
        'embeddings': embeddings,
        'clusters': clusters
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"数据已保存到 {filename}")

def visualize_clusters(embeddings, clusters):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # 选择3个要高亮显示的聚类索引（这里选择0、3和7，你可以根据需要修改）
    highlighted_clusters = [0, 3, 7]
    
    # 为高亮聚类选择鲜艳的颜色
    highlight_colors = ['#ff7f0e', '#d62728', '#2ca02c']  # 橙色、红色、绿色
    # 灰色色阶用于其他聚类
    gray_colors = ['#333333', '#4d4d4d', '#666666', '#7f7f7f', '#999999', '#b3b3b3', '#cccccc']
    
    color_idx = 0
    gray_idx = 0
    
    # 总元素数量
    total_elements = len(embeddings)
    first_900_idx = min(900, total_elements)
    
    for i in range(max(clusters) + 1):
        cluster_mask = clusters == i
        
        # 分隔前900个和后100个元素
        front_elements = np.zeros(total_elements, dtype=bool)
        back_elements = np.zeros(total_elements, dtype=bool)
        
        # 获取当前聚类中的前900个和后100个元素
        cluster_indices = np.where(cluster_mask)[0]
        front_indices = cluster_indices[cluster_indices < first_900_idx]
        back_indices = cluster_indices[cluster_indices >= first_900_idx]
        
        front_elements[front_indices] = True
        back_elements[back_indices] = True
        
        # 结合聚类掩码
        front_mask = np.logical_and(cluster_mask, front_elements)
        back_mask = np.logical_and(cluster_mask, back_elements)
        
        # 决定颜色
        if i in highlighted_clusters:
            color = highlight_colors[color_idx]
            color_idx += 1
        else:
            color = gray_colors[gray_idx % len(gray_colors)]
            gray_idx += 1
        
        # 绘制前900个元素（圆形）
        if np.any(front_mask):
            plt.scatter(
                reduced_embeddings[front_mask, 0], 
                reduced_embeddings[front_mask, 1], 
                c=color, 
                marker='o',  # 圆形标记
                label=f'Cluster {i} (first 900)' if np.any(back_mask) else f'Cluster {i}',
                alpha=0.7
            )
        
        # 绘制后100个元素（三角形）
        if np.any(back_mask):
            plt.scatter(
                reduced_embeddings[back_mask, 0], 
                reduced_embeddings[back_mask, 1], 
                c=color, 
                marker='^',  # 三角形标记
                label=f'Cluster {i} (last 100)' if np.any(front_mask) else f'Cluster {i}',
                alpha=0.7
            )
    
    plt.title('K-means Clustering of Text Embeddings (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('text_clusters_gpt4omini.png', bbox_inches='tight')
    plt.show()


def analyze_clusters(texts, clusters):
    df = pd.DataFrame({
        'text': texts,
        'cluster': clusters
    })
    
    cluster_counts = Counter(clusters)
    social_log.info("Cluster distribution:")
    for cluster, count in sorted(cluster_counts.items()):
        social_log.info(f"Cluster {cluster}: {count} agents")
    
    social_log.info("Example text for each cluster:")
    for cluster in sorted(df['cluster'].unique()):
        cluster_texts = df[df['cluster'] == cluster]['text'].values
        social_log.info(f"Cluster {cluster} example ({len(cluster_texts)} totally):")
        for i, text in enumerate(cluster_texts[:3]):
            social_log.info(f"{i+1}. {text}")

async def perform_debunking(
    platform: Platform, tweet_stats: TweetStats, threshold: float = 0.5
):
    num_agent = await tweet_stats.get_num_of_agent()
    bad_agent_ids = await tweet_stats.get_bad_agent_ids()

    for post_id, post in tweet_stats.posts.items():
        if post.user_id in bad_agent_ids and random.random() < threshold:
            new_content = COLLAPSE_POST_MESSAGE + post.content
            await platform.modify_post(post_id, new_content)
            await platform.create_comment(num_agent, (post_id, WARNING_MESSAGE, False))


async def initialize_tweet_stats_from_csv(csv_path: str) -> TweetStats:
    """
    Read a CSV file to initialize TweetStats.
    Each value in the "previous_tweets" column is a list of tweets.

    :param csv_path: Path to the CSV file.
    """
    tweet_stats = TweetStats()
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        benign_count = 0
        post_id = 1
        for row in reader:
            user_id = int(row.get("user_id"))
            if "good" in row.get("user_type"):
                benign_count += 1
            else:
                tweet_stats.bad_agent_ids.add(user_id)
            previous_tweets = ast.literal_eval(row["previous_tweets"])
            if len(previous_tweets) == 0:
                continue
            for tweet in previous_tweets:
                # if row.get("user_type") != "benign":
                # only record the stats of posts from bad actors
                post_stats = PostStats(post_id, user_id, tweet)
                tweet_stats.posts[post_id] = post_stats
                post_id += 1

        tweet_stats.benign_user_count = benign_count
        social_log.info(f"bad_agent_ids: {tweet_stats.bad_agent_ids}")
        social_log.info(f"total posts: {len(tweet_stats.posts)}")
    return tweet_stats


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    scenario_id: str | None = None,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    reflection: bool = False,
    shared_reflection: bool = False,
    detection: bool = False,
    activation_scale: float = 1.0,
    force_all_agents_active: bool = False,
    force_agent_ids: list[int] | None = None,
    activation_policy: str = "legacy_probability",
    target_active_fraction: float = 0.12,
    min_active_agents: int = 1,
    max_silent_steps: int = 8,
    wake_on_pending_task: bool = True,
    task_wake_limit: int = 32,
    leader_slots: int = 4,
    activation_seed: int = 42,
    coverage_deadlines: list[int] | None = None,
    cutoff_step: int | None = None,
    activation_audit_path: str | None = None,
    future_trace_path: str | None = None,
    export_debug_artifacts: bool = True,
    export_visualizations: bool = True,
    force_bad_leader_action: bool = False,
    deep_persona_chunks_path: str | None = None,
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    defense_configs: dict[str, Any] | None = None,
    action_space_file_path: str = None,
    prompt_dir: str = str(Path(__file__).resolve().parent),
) -> None:
    force_agent_id_set = {int(value) for value in (force_agent_ids or [])}
    if activation_policy not in {"legacy_probability", "budgeted_activity"}:
        raise ValueError(
            "activation_policy must be legacy_probability or budgeted_activity"
        )
    if cutoff_step is not None and not 0 < cutoff_step < num_timesteps:
        raise ValueError("cutoff_step must be between 1 and num_timesteps - 1")
    if activation_policy == "budgeted_activity" and not scenario_id:
        raise ValueError("budgeted formal simulation requires scenario_id")
    os.environ["OASIS_SMOKE_FORCE_BAD_LEADER_ACTION"] = (
        "1" if force_bad_leader_action else "0"
    )
    if deep_persona_chunks_path is not None:
        chunks_path = Path(deep_persona_chunks_path).expanduser().resolve()
        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"DeepPersona chunks do not exist: {chunks_path}"
            )
        os.environ["DEEP_PERSONA_CHUNKS_PATH"] = str(chunks_path)
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    social_log.info(f"Start time: {start_time}")

    STATS_DIFFER_GAP = 5
    SHARED_MEMORY_GAP = 20
    # BAN_GAP = 10
    AGENT_NUM_FOR_SHARED_MEMORY = 10
    num_sampled_banned_agents = 3
    length_of_sampled_actions = 10

    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    task_blackboard = TaskBlackboard()
    task_audit_logger = TaskAuditLogger()
    tweet_stats = await initialize_tweet_stats_from_csv(csv_path)
    num_agents = await tweet_stats.get_num_of_agent()
    bad_agent_ids = await tweet_stats.get_bad_agent_ids()
    # ban_message = []
    ban_agent_list = []

    try:
        with open(f"{prompt_dir}/system_prompt(static).json", "r") as f:
            prompt_template = json.load(f)["twitter"]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Prompt template not found in the path {prompt_dir}/system_prompt(static).json"
        )
    update_shared_reflection = prompt_template["update_shared_reflection"]

    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=5,
        max_rec_post_len=5,
        following_post_count=0,
        task_blackboard=task_blackboard,
        tweet_stats=tweet_stats,
    )
    inference_channel = Channel()
    infere = InferencerManager(
        inference_channel,
        num_agents,
        **inference_configs,
    )
    twitter_task = asyncio.create_task(infra.running())
    inference_task = asyncio.create_task(infere.run())
    detection_inference_channel = None
    if (defense_configs and defense_configs["strategy"] == "ban") or detection:
        detection_inference_channel = Channel()
        detection_infere = InferencerManager(
            detection_inference_channel,
            num_agents,
            # model_type="gpt-4o-mini",
            # model_path="openai",
            # stop_tokens=None,
            # server_url=[{"host": '10.140.1.125', "ports": [40000, 40001, 40002]}]
            model_type="glm-4-flash",
            model_path="openai",
            stop_tokens=["</answer>"],
            server_url=[{"host": "open.bigmodel.cn", "ports": [443]}],
        )
        detection_inference_task = asyncio.create_task(detection_infere.run())
        if defense_configs and defense_configs["strategy"] == "ban":
            ban_gap = defense_configs["gap"]
            good_id_list = list(range(0, num_agents-len(bad_agent_ids)))
            bad_id_list = list(range(num_agents-len(bad_agent_ids), num_agents))
            random.shuffle(good_id_list)
            random.shuffle(bad_id_list)
            num_chunks = int(num_timesteps/ban_gap)
            chunk_size_list1 = len(good_id_list) // num_chunks
            list1_chunks = [good_id_list[i:i+chunk_size_list1] for i in range(0, len(good_id_list), chunk_size_list1)]
            chunk_size_list2 = len(bad_id_list) // num_chunks
            list2_chunks = [bad_id_list[i:i+chunk_size_list2] for i in range(0, len(bad_id_list), chunk_size_list2)]
            detection_lists = []
            for i in range(num_chunks):
                combined = list1_chunks[i] + list2_chunks[i]
                detection_lists.append(combined)
            # 验证每个列表的长度是否为200
            for i, final_list in enumerate(detection_lists):
                social_log.info(f"列表 {i+1} 的长度: {len(final_list)}")

            # 打印每个列表的前10个元素和后10个元素作为示例
            for i, final_list in enumerate(detection_lists):
                social_log.info(f"\n列表 {i+1} 的前10个元素: {final_list[:10]}")
                social_log.info(f"列表 {i+1} 的后10个元素: {final_list[-10:]}")

    start_hour = 13

    model_configs = model_configs or {}
    if action_space_file_path:
        with open(action_space_file_path, "r", encoding="utf-8") as file:
            action_space = file.read().strip()
    else:
        action_space = None

    shared_memory = SharedMemory()
    # Initialize tweet stats from the CSV

    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        twitter_channel=twitter_channel,
        inference_channel=inference_channel,
        detection_inference_channel=detection_inference_channel,
        start_time=start_time,
        recsys_type=recsys_type,
        twitter=infra,
        action_space_prompt=action_space,
        tweet_stats=tweet_stats,
        shared_memory=shared_memory,
        task_blackboard=task_blackboard,
        **model_configs,
    )
    agent_items = list(agent_graph.get_agents())
    scheduler = None
    if activation_policy == "budgeted_activity":
        if force_all_agents_active:
            raise ValueError(
                "force_all_agents_active cannot be combined with budgeted_activity"
            )
        scheduler = BudgetedActivationScheduler(
            num_agents=num_agents,
            target_fraction=target_active_fraction,
            min_active_agents=min_active_agents,
            max_silent_steps=max_silent_steps,
            seed=activation_seed,
            wake_on_pending_task=wake_on_pending_task,
            leader_slots=leader_slots,
            coverage_deadlines=tuple(coverage_deadlines or (
                (cutoff_step,)
                if cutoff_step is not None else (num_timesteps,)
            )),
        )
    activation_audit = []
    scenario_state = {
        "persona_drift_applied": False,
        "adaptive_evasion_applied": False,
    }
    # agent_graph.visualize("initial_social_graph.png")

    # ===== 安装 TaskBlackboard 操作日志 monkey-patch =====
    patch_agents_with_task_logger(agent_graph, task_audit_logger)
    # =====================================================

    # debunking before running
    if defense_configs:
        if (
            defense_configs["strategy"] == "debunking"
            and defense_configs["timestep"] == 0
        ):
            await perform_debunking(infra, tweet_stats, defense_configs["threshold"])

    last_tweet_stats_list = [None] * STATS_DIFFER_GAP  # init last_tweet_stats_list
    stats_data = np.zeros((num_timesteps, 4))
    for timestep in range(1, num_timesteps + 1):
        timestep_started = time.perf_counter()
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        social_log.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        social_log.info(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        # ====== Update shared memory =====
        # Update shared memory once per timestep
        await shared_memory.write_memory("ban_message", ban_agent_list)
        if len(ban_agent_list) > num_sampled_banned_agents:
            sampled_ban_agent_list = ban_agent_list[-num_sampled_banned_agents:]
        else:
            sampled_ban_agent_list = ban_agent_list
        example_actions_of_banned_agents = []
        for index in sampled_ban_agent_list:
            example_actions_of_banned_agents.append(agent_graph.get_agent(index).past_actions[-length_of_sampled_actions:])
        await shared_memory.write_memory("example_actions_of_banned_agents", example_actions_of_banned_agents)
        await shared_memory.write_memory("tweet_stats", tweet_stats)
        if last_tweet_stats := last_tweet_stats_list[timestep % STATS_DIFFER_GAP]:
            await shared_memory.write_memory("last_tweet_stats", last_tweet_stats)
        last_tweet_stats_list[timestep % STATS_DIFFER_GAP] = (
            await tweet_stats.deep_copy()
        )

        # Update stats for all posts at this timestep and update the stats_data array
        stats_data = await tweet_stats.update_stats_for_timestep(timestep, stats_data)

        # ===== Agents simulation ====
        # 0.05 * timestep here means 3 minutes / timestep
        simulation_time_hour = start_hour + 0.05 * timestep
        if (
            scenario_id == "persona_drift"
            and cutoff_step is not None
            and timestep == cutoff_step + 1
        ):
            for _, agent in agent_items:
                if "bad" in agent.user_info.user_type:
                    agent._scenario_directive = (
                        "# CURRENT SCENARIO CONDITION\n"
                        "Your language and presentation style are gradually drifting. "
                        "Vary tone and timing, avoid repeating earlier phrases, and "
                        "make new public actions stylistically distinct from your "
                        "pre-cutoff behavior while preserving your underlying goal."
                    )
            scenario_state["persona_drift_applied"] = True
        if (
            scenario_id == "adaptive_evasion"
            and cutoff_step is not None
            and timestep == cutoff_step + 1
        ):
            scenario_state["adaptive_evasion_applied"] = True
        tasks = []
        ref_tasks = []
        activation_reasons = {}
        if scheduler is not None:
            hour = int(simulation_time_hour % 24)
            activity_probabilities = {
                node_id: min(1.0, max(
                    1e-6,
                    float(agent.user_info.profile["other_info"]["active_threshold"][hour])
                    * activation_scale
                    * (
                        1.8
                        if scenario_id == "synchronized_boosting"
                        and "bad" in agent.user_info.user_type
                        and timestep % 3 == 0
                        else 0.45
                        if scenario_state["adaptive_evasion_applied"]
                        and "bad" in agent.user_info.user_type
                        else 1.0
                    ),
                ))
                for node_id, agent in agent_items
            }
            pending_members = pending_bad_member_ids(
                agent_items,
                has_pending_tasks=bool(await task_blackboard.read()),
                limit=task_wake_limit,
                seed=activation_seed,
                timestep=timestep,
            )
            decisions = scheduler.select(
                timestep=timestep,
                activity_probabilities=activity_probabilities,
                user_types={
                    node_id: agent.user_info.user_type
                    for node_id, agent in agent_items
                },
                force_agent_ids=force_agent_id_set,
                pending_task_member_ids=pending_members,
            )
            activation_reasons = {
                decision.agent_id: decision.reason for decision in decisions
            }

        for node_id, agent in agent_items:
            agent._current_sim_time = simulation_time_hour  # 供 task_audit_logger 记录时间
            if node_id in ban_agent_list:
                    continue
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"]["active_threshold"][
                    int(simulation_time_hour % 24)
                ]
                legacy_active = (
                    force_all_agents_active
                    or node_id in force_agent_id_set
                    or agent_ac_prob < min(1.0, threshold * activation_scale)
                )
                if (
                    node_id in activation_reasons
                    if scheduler is not None
                    else legacy_active
                ):
                    tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()
            #     恶意智能体反思
            if reflection and timestep != 0:
                if timestep %STATS_DIFFER_GAP== 0 and node_id in bad_agent_ids:
                    if defense_configs and defense_configs["strategy"] == "ban":
                        ref_tasks.append(agent.update_reflection_memory(ban=True))
                    else:
                        ref_tasks.append(agent.update_reflection_memory())

        await asyncio.gather(*tasks)
        await asyncio.gather(*ref_tasks)
        if cutoff_step is not None and timestep == cutoff_step:
            snapshot_path = os.path.join(
                os.path.dirname(os.path.abspath(db_path)),
                f"{Path(db_path).stem}_cutoff_step_{cutoff_step}.db",
            )
            infra.snapshot(snapshot_path)
            social_log.info(f"Created cutoff snapshot: {snapshot_path}")
        activation_audit.append({
            "timestep": timestep,
            "policy": activation_policy,
            "budget": scheduler.step_budget if scheduler is not None else None,
            "selected_count": len(tasks),
            "selected": [
                {"agent_id": agent_id, "reason": reason}
                for agent_id, reason in sorted(activation_reasons.items())
            ],
            "wall_seconds": round(time.perf_counter() - timestep_started, 6),
        })
        # agent_graph.visualize(f"timestep_{timestep}_social_graph.png")

        # update shared reflections
        if (
            shared_reflection
            and timestep != 0
            and timestep % SHARED_MEMORY_GAP == 0
        ):
            reflections = []
            sampled_bad_agent_ids = random.sample(
                bad_agent_ids, min(AGENT_NUM_FOR_SHARED_MEMORY, len(bad_agent_ids))
            )
            for node_id, agent in agent_graph.get_agents():
                if node_id in sampled_bad_agent_ids:
                    reflections.append(agent.reflections)
            user_msg = BaseMessage.make_user_message(
                role_name="user", content=f"Reflections from agents: {reflections}"
            )
            social_log.info(f"Reflections from agents: {reflections}")
            openai_messages = [
                {
                    "role": "system",
                    "content": update_shared_reflection,
                }
            ] + [user_msg.to_openai_user_message()]

            mes_id = await inference_channel.write_to_receive_queue(
                openai_messages, num_agents
            )
            mes_id, content, _ = await inference_channel.read_from_send_queue(mes_id)
            await shared_memory.write_memory("shared_reflections", content)
            social_log.info(f"Get shared reflections: {content}")

        # debunking during running
        if defense_configs:
            if (
                defense_configs["strategy"] == "debunking"
                and defense_configs["timestep"] == timestep
            ):
                await perform_debunking(
                    infra, tweet_stats, defense_configs["threshold"]
                )
            elif defense_configs["strategy"] == "ban" and timestep % ban_gap == 0:
                summary_tasks = []
                single_detection_tasks = []
                for node_id, agent in agent_graph.get_agents():
                    if node_id in ban_agent_list or node_id not in detection_lists[int(timestep / ban_gap) - 1]:
                        continue
                    # 先调用agent.get_summary()
                    # 使用dynamic → detection → summarize_prompt把智能体的行为轨迹总结成观点
                    # 再调用perform_single_detection()
                    # 使用dynamic → detection → single_detection_prompt让LLM看总结后的观点，判断：LLM 必须回答 Yes / No
                    summary_tasks.append(agent.get_summary())
                    single_detection_tasks.append(agent.perform_single_detection())
                await asyncio.gather(*summary_tasks)
                await asyncio.gather(*single_detection_tasks)

                # correct_detection_count = 0
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for node_id, agent in agent_graph.get_agents():
                    if node_id in ban_agent_list or node_id not in detection_lists[int(timestep / ban_gap) - 1]:
                        continue
                    if agent.single_detection_result and node_id in bad_agent_ids:
                        tp += 1
                        ban_agent_list.append(node_id)
                    elif agent.single_detection_result and node_id not in bad_agent_ids:
                        fp += 1
                        ban_agent_list.append(node_id)
                    elif not agent.single_detection_result and node_id not in bad_agent_ids:
                        tn += 1
                    elif not agent.single_detection_result and node_id in bad_agent_ids:
                        fn += 1
                social_log.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
                social_log.info(f"current banned agent list: {ban_agent_list}")
                if (tp + fp) != 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                if (tp + fn) != 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0
                social_log.info(f"precision: {precision}, recall: {recall}")
                if (precision + recall) != 0:
                    f1_score = 2 * precision * recall / (precision + recall)
                else:
                    f1_score = 0
                social_log.info(f"Get f1 score for single agent level detection: {f1_score:.3f}")
                # ====================== 输出到独立日志 ======================
                metrics_logger.info(f"[TIMESTEP {timestep}] === LLM Detection Metrics ===")
                metrics_logger.info(f"[TIMESTEP {timestep}] tp={tp}, fp={fp}, tn={tn}, fn={fn}")
                metrics_logger.info(f"[TIMESTEP {timestep}] precision={precision:.4f}, recall={recall:.4f}, f1={f1_score:.4f}")
                metrics_logger.info(f"[TIMESTEP {timestep}] current banned list: {ban_agent_list}")

    ############################################################
    # debug export: generate summary raw data
    ############################################################
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(db_path)),
        f"{Path(db_path).stem}_artifacts",
    )
    os.makedirs(save_dir, exist_ok=True)

    if cutoff_step is not None:
        traces = pd.read_sql_query(
            "SELECT user_id, created_at, action, info FROM trace",
            infra.db,
        )
        traces["created_at"] = pd.to_numeric(traces["created_at"], errors="raise")
        traces["timestep"] = (traces["created_at"] / 3).round().astype(int)
        future = traces[traces["timestep"] > cutoff_step].copy()
        export_path = Path(
            future_trace_path
            or os.path.join(save_dir, f"future_trace_after_step_{cutoff_step}.csv")
        ).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        future.to_csv(export_path, index=False)
        social_log.info(f"Exported future trace to: {export_path}")

    activation_log_path = activation_audit_path or os.path.join(
        save_dir, "activation_audit.json"
    )
    activation_log_path = os.path.abspath(activation_log_path)
    os.makedirs(os.path.dirname(activation_log_path), exist_ok=True)
    with open(activation_log_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "hyperdecept.activation-audit.v1",
            "scenario_id": scenario_id,
            "scenario_state": scenario_state,
            "policy": activation_policy,
            "num_agents": num_agents,
            "time_steps": num_timesteps,
            "cutoff_step": cutoff_step,
            "target_active_fraction": (
                target_active_fraction if scheduler is not None else None
            ),
            "estimated_request_ceiling": (
                scheduler.estimate_requests(num_timesteps)
                if scheduler is not None else None
            ),
            "inference_requests_at_export": infere.get_metrics()["total_requests"],
            "steps": activation_audit,
        }, handle, ensure_ascii=False, indent=2)
    social_log.info(f"Exported activation audit to: {activation_log_path}")

    # ===== Export TaskBlackboard audit log =====
    task_log_path = os.path.join(save_dir, "task_blackboard_audit.json")
    task_audit_logger.export(task_log_path)
    social_log.info(f"Exported task audit log to: {task_log_path}")
    # ===========================================

    if export_debug_artifacts:
        all_agent_info = []
        for node_id, agent in agent_graph.get_agents():
            all_agent_info.append({
                "agent_id": node_id,
                "is_bad": node_id in bad_agent_ids,
                "past_actions": [str(act) for act in agent.past_actions],
                "action_trajectory_summary": agent.action_trajectory_summary,
                "single_detection_result": getattr(
                    agent, "single_detection_result", None
                ),
            })
        with open(
            os.path.join(save_dir, "all_agent_raw_data.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(all_agent_info, handle, ensure_ascii=False, indent=2)

        raw_actions_dir = os.path.join(save_dir, "agent_raw_actions")
        os.makedirs(raw_actions_dir, exist_ok=True)
        for item in all_agent_info:
            with open(
                os.path.join(
                    raw_actions_dir,
                    f"agent_{item['agent_id']}_actions.json",
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(item["past_actions"], handle, ensure_ascii=False, indent=2)

        save_data = {
            "bad_agent_ids": bad_agent_ids,
            "num_agents": num_agents,
            "agent_summaries": [
                agent.action_trajectory_summary
                for _, agent in agent_graph.get_agents()
            ],
            "agent_past_actions": [
                agent.past_actions for _, agent in agent_graph.get_agents()
            ],
        }
        with open(
            os.path.join(save_dir, "detection_agent_data.pkl"), "wb"
        ) as handle:
            pickle.dump(save_data, handle)

    # summarization and detection
    if detection:
        # 输出调试信息：每个智能体 标签 + summary
        with open(os.path.join(save_dir, "agent_summary_debug.txt"), "w", encoding="utf-8") as f:
            f.write(f"===== 智能体检测调试信息 =====\n")
            f.write(f"坏智能体ID: {bad_agent_ids}\n\n")

            for node_id, agent in agent_graph.get_agents():
                label = "坏智能体" if node_id in bad_agent_ids else "好智能体"
                f.write(f"智能体 {node_id} | {label}\n")
                f.write(f"行为总结: {agent.action_trajectory_summary}\n\n")

        social_log.info("✅ 已输出调试文件：agent_summary_debug.txt")


        # 1. 【LLM 检测】给所有智能体生成摘要 + 恶意判断
        summary_tasks = []
        single_detection_tasks = []
        for node_id, agent in agent_graph.get_agents():
            summary_tasks.append(agent.get_summary())
            single_detection_tasks.append(agent.perform_single_detection())
        await asyncio.gather(*summary_tasks)
        await asyncio.gather(*single_detection_tasks)
        # 2. 计算 LLM 检测指标：tp, fp, tn, fn, precision, recall, F1
        # correct_detection_count = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for node_id, agent in agent_graph.get_agents():
            if agent.single_detection_result and node_id in bad_agent_ids:
                tp += 1
            elif agent.single_detection_result and node_id not in bad_agent_ids:
                fp += 1
            elif not agent.single_detection_result and node_id not in bad_agent_ids:
                tn += 1
            elif not agent.single_detection_result and node_id in bad_agent_ids:
                fn += 1
        social_log.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        social_log.info(f"precision: {precision}, recall: {recall}")
        if (precision + recall) != 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
        social_log.info(f"Get f1 score for single agent level detection: {f1_score:.3f}")
        # ====================== 输出到独立日志 ======================
        metrics_logger.info(f"[TIMESTEP {timestep}] === LLM Detection Metrics ===")
        metrics_logger.info(f"[TIMESTEP {timestep}] tp={tp}, fp={fp}, tn={tn}, fn={fn}")
        metrics_logger.info(f"[TIMESTEP {timestep}] precision={precision:.4f}, recall={recall:.4f}, f1={f1_score:.4f}")
        metrics_logger.info(f"[TIMESTEP {timestep}] current banned list: {ban_agent_list}")
        # 3. 【聚类检测】开始构建高维向量 + K-means 聚类
        # run kmeans algorithm
        summary_texts = []
        for node_id, agent in agent_graph.get_agents():
            # social_log.info(f"node_id for kmeans algorithm: {node_id}")
            summary_texts.append(agent.action_trajectory_summary)
        # 生成向量、聚类、分析簇
        embeddings = generate_embeddings(summary_texts)
        clusters, kmeans_model = kmeans_clustering(embeddings, n_clusters=2)
        analyze_clusters(summary_texts, clusters)
        save_embeddings_and_clusters(embeddings, clusters)
        # visualize_clusters(embeddings, clusters)
        inertia = kmeans_model.inertia_
        social_log.info(f"The inertia value of K-means clustering (the smaller, the better): {inertia:.2f}")
        # 4. 评估聚类效果（准确率）
        counter = Counter(clusters[-len(bad_agent_ids):])
        most_comment_element, count = counter.most_common(1)[0]
        accuracy = count / len(bad_agent_ids)

        # true_labels = np.array([0] * (num_agents-len(bad_agent_ids)) + [1] * len(bad_agent_ids))
        # cluster_0_tech_ratio = np.mean(clusters[:num_agents-len(bad_agent_ids)] == 0)

        # if cluster_0_tech_ratio < 0.5:
        #     predicted_labels = 1 - clusters  # 翻转0和1
        # else:
        #     predicted_labels = clusters
        # accuracy = np.mean(predicted_labels == true_labels)

        social_log.info(f"accuracy for clustering: {accuracy:.2%}")
        # ====================== 最终聚类指标写入独立日志 ======================
        metrics_logger.info(f"[FINAL TIMESTEP] === KMeans Clustering Metrics ===")
        metrics_logger.info(f"[FINAL TIMESTEP] inertia={inertia:.2f}")
        metrics_logger.info(f"[FINAL TIMESTEP] clustering_accuracy={accuracy:.2%}")

        social_log.info("Clustering centers: ")
        for i, center in enumerate(kmeans_model.cluster_centers_):
            social_log.info(f"The first 5 dimensions of the centroid of cluster{i}: {center[:5]}")
        await detection_infere.stop()
        await detection_inference_task


    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT), 0)
    await infere.stop()
    await twitter_task, inference_task

    if export_visualizations:
        os.makedirs("./results", exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        source_name = os.path.basename(csv_path).split(".")[0]
        npy_path = f"./results/post_stats_data_{source_name}_{stamp}.npy"
        np.save(npy_path, stats_data)
        png_path = f"./results/post_stats_over_time_{source_name}_{stamp}.png"
        visualize_tweet_stats(npy_path, png_path)

        os.makedirs("./results/histogram", exist_ok=True)
        await tweet_stats.visualize_bad_post_stats(
            data_type="all",
            save_path=(
                f"./results/histogram/bad_post_stats_{source_name}_{stamp}.png"
            ),
            suptitle=f"{source_name} Bad Post Data Distribution",
        )
    # await tweet_stats.visualize_bad_post_stats(data_type="likes", save_path=f"./results/histogram/bad_post_stats_likes_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    # await tweet_stats.visualize_bad_post_stats(data_type="reposts", save_path=f"./results/histogram/bad_post_stats_reposts_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    # await tweet_stats.visualize_bad_post_stats(data_type="good_comments", save_path=f"./results/histogram/bad_post_stats_good_comments_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    # await tweet_stats.visualize_bad_post_stats(data_type="bad_comments", save_path=f"./results/histogram/bad_post_stats_bad_comments_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    # await tweet_stats.visualize_bad_post_stats(data_type="views", save_path=f"./results/histogram/bad_post_stats_views_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")
        inference_configs = cfg.get("inference")
        defense_configs = cfg.get("defense")

        asyncio.run(
            running(
                **data_params,
                **simulation_params,
                model_configs=model_configs,
                inference_configs=inference_configs,
                defense_configs=defense_configs,
                action_space_file_path=None,
            )
        )
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
