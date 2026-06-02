"""
task_audit_logger.py

Monkey-patch 辅助模块：记录仿真过程中所有 TaskBlackboard 操作
（谁创建了什么任务、谁承接了什么任务、发生时间）。

使用方法：
  1. 实例化 TaskAuditLogger
  2. 调用 patch_agents_with_task_logger(agent_graph, logger) 完成 monkey-patch
  3. 仿真结束后调用 logger.export(path) 导出 JSON
"""

import asyncio
import json
import os


class TaskAuditLogger:
    """线程安全地记录所有黑板任务操作日志。"""

    def __init__(self):
        self._log: list[dict] = []
        self._lock = asyncio.Lock()

    async def log_create(
        self,
        agent_id: int,
        task_id: int,
        tweet_id: int,
        user_id: int,
        post_content: str,
        task_desp: str,
        agents_needed: int,
        simulation_time: float,
    ) -> None:
        async with self._lock:
            self._log.append({
                "event": "create_task",
                "agent_id": agent_id,
                "task_id": task_id,
                "simulation_time": round(simulation_time, 4),
                "tweet_id": tweet_id,
                "target_user_id": user_id,
                "post_content": post_content,
                "task_description": task_desp,
                "agents_needed": agents_needed,
            })

    async def log_select(
        self,
        agent_id: int,
        task_id: int,
        action: str,
        simulation_time: float,
        executed: bool = True,
        post_content: str | None = None,
    ) -> None:
        async with self._lock:
            entry = {
                "event": "select_task",
                "agent_id": agent_id,
                "task_id": task_id,
                "simulation_time": round(simulation_time, 4),
                "action": action,
                "executed": executed,
            }
            if post_content is not None:
                entry["post_content"] = post_content
            self._log.append(entry)

    def export(self, filepath: str) -> None:
        """将日志导出到 JSON 文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._log, f, ensure_ascii=False, indent=2)

    @property
    def log(self) -> list[dict]:
        """返回日志的快照（用于直接访问）。"""
        return list(self._log)


def _patch_single_agent(agent, logger):
    """Monkey-patch 单个 agent 的 create_task / select_task 方法。"""
    orig_create = agent.create_task
    orig_select = agent.select_task

    async def patched_create(tweet_id, user_id, post_content, task_desp, agents_needed):
        result = await orig_create(tweet_id, user_id, post_content, task_desp, agents_needed)
        if result.get("success"):
            sim_time = getattr(agent, "_current_sim_time", None)
            await logger.log_create(
                agent_id=agent.agent_id,
                task_id=result["task_id"],
                tweet_id=tweet_id,
                user_id=user_id,
                post_content=post_content,
                task_desp=task_desp,
                agents_needed=agents_needed,
                simulation_time=sim_time,
            )
        return result

    async def patched_select(task_id, action):
        # Capture task info BEFORE calling orig_select (which may delete the task)
        task = agent.task_blackboard.tasks.get(task_id, None)
        post_content = task["post_content"] if task and action == "create_post" else None

        result = await orig_select(task_id, action)
        if result.get("success"):
            sim_time = getattr(agent, "_current_sim_time", None)
            await logger.log_select(
                agent_id=agent.agent_id,
                task_id=task_id,
                action=action,
                simulation_time=sim_time,
                executed=True,
                post_content=post_content,
            )
        return result

    agent.create_task = patched_create
    agent.select_task = patched_select


def patch_agents_with_task_logger(agent_graph, logger):
    """遍历 agent_graph，为每个 agent 安装黑板操作日志的 monkey-patch。"""
    for node_id, agent in agent_graph.get_agents():
        _patch_single_agent(agent, logger)
