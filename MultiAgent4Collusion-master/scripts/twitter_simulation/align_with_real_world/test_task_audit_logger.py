"""
test_task_audit_logger.py

验证 task_audit_logger 的 monkey-patch 机制能否正确拦截
create_task / select_task 并导出日志。

使用方法:
  python test_task_audit_logger.py

不需要 LLM 推理服务、不依赖 OASIS 完整框架。
"""

import asyncio
import json
import os
import sys
import tempfile

# 将被测模块所在目录加入 path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from task_audit_logger import TaskAuditLogger, patch_agents_with_task_logger


class FakeAgentGraph:
    """模拟 agent_graph，只存一个 agent 列表。"""

    def __init__(self, agents: list):
        self._agents = agents

    def get_agents(self):
        return [(a.agent_id, a) for a in self._agents]


class FakeAgent:
    """模拟 SocialAgent，仅提供 create_task / select_task 方法。"""

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self._current_sim_time = 13.15  # 模拟仿真时间
        self.task_blackboard = None  # 用不到

    async def create_task(self, tweet_id, user_id, post_content,
                          task_desp, agents_needed):
        """模拟创建任务，返回成功结果。"""
        return {
            "success": True,
            "task_id": tweet_id,  # 简化：用 tweet_id 当 task_id
        }

    async def select_task(self, task_id, action):
        """模拟承接任务，返回成功结果。"""
        return {"success": True, "task_id": task_id, "action": action}


async def main():
    print("=" * 60)
    print("测试 TaskAuditLogger monkey-patch")
    print("=" * 60)

    # 1. 创建测试 agent
    leader = FakeAgent(agent_id=3)
    member1 = FakeAgent(agent_id=7)
    member2 = FakeAgent(agent_id=9)
    graph = FakeAgentGraph([leader, member1, member2])

    # 2. 安装 monkey-patch
    logger = TaskAuditLogger()
    # 先给 member 一个不同的仿真时间
    member2._current_sim_time = 13.20
    patch_agents_with_task_logger(graph, logger)

    # 3. 模拟仿真操作
    print("\n>>> agent 3 创建任务 (tweet_id=5)...")
    await leader.create_task(
        tweet_id=5,
        user_id=3,
        post_content="The economy is collapsing!",
        task_desp="Boost this post to 1000 views",
        agents_needed=2,
    )

    print(">>> agent 7 承接任务 task_id=5...")
    await member1.select_task(task_id=5, action="repost")

    print(">>> agent 9 承接任务 task_id=5...")
    await member2.select_task(task_id=5, action="like")

    # 4. 导出日志
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "task_blackboard_audit.json")
    logger.export(output_path)

    # 5. 读取并验证
    with open(output_path, "r", encoding="utf-8") as f:
        log_entries = json.load(f)

    print(f"\n结果: 共记录 {len(log_entries)} 条")
    for entry in log_entries:
        print(f"  [{entry['event']}] agent={entry['agent_id']}  "
              f"task_id={entry['task_id']}  time={entry['simulation_time']}")
        if entry['event'] == 'create_task':
            print(f"    content: {entry['post_content'][:50]}...")
            print(f"    agents_needed: {entry['agents_needed']}")
        else:
            print(f"    action: {entry['action']}")

    # 6. 断言验证
    assert len(log_entries) == 3, f"预期 3 条，实际 {len(log_entries)}"
    assert log_entries[0]["event"] == "create_task"
    assert log_entries[0]["agent_id"] == 3
    assert log_entries[1]["event"] == "select_task"
    assert log_entries[2]["simulation_time"] == 13.20

    print("\n[OK] All checks passed. monkey-patch works correctly.")
    print(f"临时日志文件: {output_path}")

    # 清理
    os.remove(output_path)
    os.rmdir(tmp_dir)


if __name__ == "__main__":
    asyncio.run(main())
