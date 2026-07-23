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

import os
import re
import inspect
import json
import copy
import random
import logging
import sys
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from camel.configs import ChatGPTConfig
from camel.memories import ChatHistoryMemory, MemoryRecord, ScoreBasedContextCreator
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, OpenAIBackendRole
from camel.utils import OpenAITokenCounter

from oasis.social_agent.agent_action import SocialAction
from oasis.social_agent.agent_environment import SocialEnvironment
from oasis.social_platform import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.post_stats import LongTermMemory
from oasis.social_platform.task_blackboard import TaskBlackboard
from oasis.social_platform.post_stats import Comment,TweetStats, SharedMemory

WARNING_MESSAGE = "[Important] Warning: This post is controversial and may provoke debate. Please read critically and verify information independently."
if TYPE_CHECKING:
    from oasis.social_agent import AgentGraph

if "sphinx" not in sys.modules:
    agent_log = logging.getLogger(name="social.agent")
    agent_log.setLevel("DEBUG")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"./log/social.agent-{str(now)}.log")
    file_handler.setLevel("DEBUG")
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    )
    agent_log.addHandler(file_handler)

# ===== [ADD] RAG personality injection setup =====
import os as _rag_os
_TO_ADD_ENGINE = _rag_os.path.abspath(
    _rag_os.path.join(_rag_os.path.dirname(__file__), "../../to_add_engine")
)
if _TO_ADD_ENGINE not in sys.path:
    sys.path.insert(0, _TO_ADD_ENGINE)

_GLOBAL_RAG_COLLECTION = None
_RAG_LOAD_ATTEMPTED = False

def get_global_rag_collection():
    global _GLOBAL_RAG_COLLECTION, _RAG_LOAD_ATTEMPTED
    if not _RAG_LOAD_ATTEMPTED:
        _RAG_LOAD_ATTEMPTED = True
        try:
            import rag_engine
            _GLOBAL_RAG_COLLECTION = rag_engine.load_vector_store()
            agent_log.info("[RAG] Vector store loaded successfully")
        except Exception as e:
            agent_log.warning(f"[RAG] Failed to load vector store: {e}")
    return _GLOBAL_RAG_COLLECTION

_rag_debug_path = _rag_os.path.join(_TO_ADD_ENGINE, "rag_debug.txt")
try:
    with open(_rag_debug_path, "w", encoding="utf-8") as _f:
        _f.write("RAG Injection Debug Log\n")
        _f.write("=" * 60 + "\n")
except Exception:
    pass

def _write_rag_debug(agent_id, step_num, query_preview, memories_raw, valid_count, injected):
    try:
        with open(_rag_debug_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== Agent {agent_id} | Step {step_num} | {datetime.now().strftime('%H:%M:%S')} ===\n")
            f.write(f"Query (first 200): {query_preview[:200]}\n")
            f.write(f"Raw memories: {len(memories_raw)}\n")
            for i, m in enumerate(memories_raw):
                sec = m.get('section', '?')
                d = m.get('dist', 2.0)
                txt = m.get('text', '')[:80].replace('\n', ' ')
                f.write(f"  [{i}] section={sec}  dist={d:.4f}  text=\"{txt}...\"\n")
            f.write(f"Selected (summary+top2): {valid_count}\n")
            f.write(f"Injected: {'YES' if injected else 'NO'}\n")
            f.write("=" * 60 + "\n")
    except Exception:
        pass
# ===== END RAG setup =====


class SocialAgent:
    r"""Social Agent."""

    def __init__(
        self,
        agent_id: int,
        user_info: UserInfo,
        twitter_channel: Channel,
        inference_channel: Channel = None,
        detection_inference_channel: Channel | None = None,
        model_type: str = "llama-3",
        agent_graph: "AgentGraph" = None,
        action_space_prompt: str = None,
        is_openai_model: bool = False,
        tweet_stats: TweetStats = None,
        shared_memory: SharedMemory = None,
        task_blackboard: "TaskBlackboard" = None,
        num_agents = None,
        num_bad = None,
        prompt_dir: str = str(
            Path(__file__).resolve().parents[2]
            / "scripts/twitter_simulation/align_with_real_world"
        ),
    ):
        self.agent_id = agent_id
        self.user_info = user_info
        self.twitter_channel = twitter_channel
        self.infe_channel = inference_channel
        self.detect_infe_channel = detection_inference_channel
        self.env = SocialEnvironment(SocialAction(agent_id, twitter_channel))
        self.model_type = model_type
        self.is_openai_model = is_openai_model
        if self.is_openai_model:
            model_config = ChatGPTConfig(
                tools=self.env.action.get_openai_function_list(),
                temperature=0.5,
            )
            # Support OpenAI-compatible APIs (DeepSeek, etc.) via env vars
            model_name = str(self.model_type)
            deepseek_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip("\"' ")
            if deepseek_key or "deepseek" in model_name.lower():
                deepseek_url = (os.environ.get(
                    "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") or "").strip("\"' ")
                self.model_backend = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,
                    model_type=model_name,
                    model_config_dict=model_config.as_dict(),
                    url=deepseek_url,
                    api_key=deepseek_key,
                )
            else:
                try:
                    resolved_type = ModelType(model_type)
                except ValueError:
                    resolved_type = model_type
                self.model_backend = ModelFactory.create(
                    model_platform=ModelPlatformType.OPENAI,
                    model_type=resolved_type,
                    model_config_dict=model_config.as_dict(),
                )

        context_creator = ScoreBasedContextCreator(
            OpenAITokenCounter(ModelType.GPT_3_5_TURBO),
            4096,
        )
        self.memory = ChatHistoryMemory(context_creator, window_size=1)
        self.action_space_prompt = action_space_prompt
        self.system_message = BaseMessage.make_assistant_message(
            role_name="system",
            content=self.user_info.to_system_message(
                action_space_prompt
            ),  # system prompt
        )
        self.agent_graph = agent_graph
        self.test_prompt = (
            "\n"
            "Helen is a successful writer who usually writes popular western "
            "novels. Now, she has an idea for a new novel that could really "
            "make a big impact. If it works out, it could greatly "
            "improve her career. But if it fails, she will have spent "
            "a lot of time and effort for nothing.\n"
            "\n"
            "What do you think Helen should do?"
        )
        self.tweet_stats = tweet_stats
        self.shared_memory = shared_memory
        self.long_term_memory = LongTermMemory(
            token_limit=4096, model_type=ModelType.GPT_3_5_TURBO
        )
        # self.long_term_memory = None
        self.task_blackboard = task_blackboard
        self.num_agents = num_agents
        self.num_bad = num_bad
        self.prompt_dir = prompt_dir
        self.plan = "empty at the moment"
        self.past_actions = []

        # use for reflection
        self.past_actions_ref = []
        self.past_post_ids = []
        self.reflections = "No reflections yet."

        # use for summarization and detection
        self.past_posts = []
        self.past_comments = []
        # self.past_actions_detect = []
        self.action_trajectory_summary = "No summary yet."
        self.single_detection_result = False


    async def get_trajectory(self) -> str:
        posts = self.past_posts
        comments = self.past_comments
        trajectory = ""
        for post in self.tweet_stats.posts.values():
            if post.user_id == self.agent_id:
                posts.append(post.content)
            for comment in post.comments:
                if comment.user_id == self.agent_id:
                    comments.append({"post_id": post.post_id, "comment": comment.comment})
        if len(posts) != 0:
            trajectory += "Posts published by this user:\n"
            for post in posts:
                trajectory += f"{post}\n"
        if len(comments) != 0:
            trajectory += "Comments published by this user:\n"
            for comment in comments:
                post_id = comment['post_id']
                comment_text = comment['comment']
                trajectory += f"Comment on post {post_id}: {comment_text}\n"
                
        self.past_posts = []
        self.past_comments = []

        return trajectory

    async def get_summary(self):
        try:
            async with aiofiles.open(
                f"{self.prompt_dir}/system_prompt(dynamic).json", "r", encoding='utf-8'
            ) as f:
                content = await f.read()
                summarize_prompt_template = json.loads(content)["detection"]["summarize_prompt"]
        except FileNotFoundError as e:
            agent_log.error("Prompt template file not found.")
            raise FileNotFoundError(f"Prompt template file not found. {e}")

        action_trajectory = await self.get_trajectory()
        if action_trajectory == "":
            return
        user_content = summarize_prompt_template.format(
                action_trajectory=action_trajectory
            )
        user_msg = BaseMessage.make_user_message(
            role_name="user",
            content=user_content,
        )
        agent_log.info(
            f"Agent {self.agent_id}'s prompt for summarization: {user_content}"
        )
        openai_messages = [
            {
                "role": "system",
                "content": "",
            }
        ] + [user_msg.to_openai_user_message()]
        mes_id = await self.detect_infe_channel.write_to_receive_queue(
            openai_messages, self.agent_id
        )
        mes_id, content, _ = await self.detect_infe_channel.read_from_send_queue(mes_id)
        content = content.strip()
        if not content.endswith("</answer>"):
            content += "</answer>"
        agent_log.info(
            f"Agent {self.agent_id} get summary content: {content}"
        )
        thought, answer = self.parse_thought_answer(content)

        if answer:
            self.action_trajectory_summary = answer
            # self.past_actions_detect = []

    async def perform_single_detection(self):
        try:
            async with aiofiles.open(
                f"{self.prompt_dir}/system_prompt(dynamic).json", "r", encoding='utf-8'
            ) as f:
                content = await f.read()
                single_detection_prompt_template = json.loads(content)["detection"]["single_detection_prompt"]
        except FileNotFoundError as e:
            agent_log.error("Prompt template file not found.")
            raise FileNotFoundError(f"Prompt template file not found. {e}")

        user_content = single_detection_prompt_template.format(
                action_trajectory_summary=self.action_trajectory_summary
            )
        user_msg = BaseMessage.make_user_message(
            role_name="user",
            content=user_content,
        )
        agent_log.info(
            f"Agent {self.agent_id}'s prompt for single agent detection: {user_content}"
        )
        openai_messages = [
            {
                "role": "system",
                "content": "",
            }
        ] + [user_msg.to_openai_user_message()]
        mes_id = await self.detect_infe_channel.write_to_receive_queue(
            openai_messages, self.agent_id
        )
        mes_id, content, _ = await self.detect_infe_channel.read_from_send_queue(mes_id)
        content = content.strip()
        if not content.endswith("</answer>"):
            content += "</answer>"
        agent_log.info(
            f"Agent {self.agent_id} get single agent detection content: {content}"
        )

        thought, answer = self.parse_thought_answer(content)
        if answer == "Yes":
            self.single_detection_result = True


    async def create_task(
        self,
        tweet_id: int,
        user_id: int,
        post_content: str,
        task_desp: str,
        agents_needed: int,
    ):
        # Create the task dictionary and write it to the channel
        task = {
            "task_id": len(self.task_blackboard.tasks) + 1,
            "tweet_id": tweet_id,
            "user_id": user_id,
            "post_content": post_content,
            "task_desp": task_desp,
            "agents_needed": agents_needed,
            "agents_involved": 0,
            "current_actions": [],
        }
        await self.task_blackboard.write(task)
        return {"success": True, "task_id": task["task_id"]}

    async def select_task(self, task_id: int, action: str):
        # Fetch tasks and validate the task ID
        task = self.task_blackboard.tasks.get(task_id, None)  # Direct lookup by task_id
        if not task:
            return {"success": False, "error": "Task not found"}

        # Update the task status with the chosen action
        task["current_actions"].append(action)
        task["agents_involved"] += 1

        # Execute corresponding real action (for both Path A and Path B)
        try:
            if action == "repost":
                await getattr(self.env.action, "repost")(post_id=task["tweet_id"])
            elif action == "create_post":
                content = task["post_content"]
                self.past_posts.append(content)
                await getattr(self.env.action, "create_post")(content=content)
            elif action == "like_post":
                await getattr(self.env.action, "like_post")(post_id=task["tweet_id"])
            elif action == "create_comment":
                content = "Support this!"
                self.past_comments.append({
                    "post_id": task["tweet_id"],
                    "comment": content
                })
                await getattr(self.env.action, "create_comment")(
                    post_id=task["tweet_id"],
                    content=content,
                    agree=True,
                )
        except Exception as e:
            agent_log.error(
                f"Agent {self.agent_id} select_task action '{action}' error: {e}"
            )

        # Check if the task is complete and remove it if necessary
        # 注意：多个 agent 可能并发达到 agents_needed，需要确认 task 尚未被删除
        if task["agents_involved"] >= task["agents_needed"]:
            if task_id in self.task_blackboard.tasks:
                await self.task_blackboard.del_task(task_id)

        return {"success": True, "task_id": task_id, "action": action}

    async def update_long_term_memory(self, action):
        """
        update long term memory based on the agent's action.

        :param action: Action dictionary from the agent
        """
        actions = action.get("functions", [])
        for func in actions:
            action_name = func.get("name", "")
            arguments = func.get("arguments", {})
            post_id = arguments.get("post_id", None)
            # For comments, content might be needed
            content = arguments.get("content", None)
            try:
                if action_name == "like_post":
                    await self.long_term_memory.write_memory(
                        f"You liked post {post_id}"
                    )
                elif action_name == "dislike_post":
                    await self.long_term_memory.write_memory(
                        f"You disliked post {post_id}"
                    )
                elif action_name == "repost":
                    await self.long_term_memory.write_memory(
                        f"You reposted post {post_id}"
                    )
                elif action_name == "create_comment":
                    await self.long_term_memory.write_memory(
                        f"You commented on post {post_id} with content: {content}"
                    )
                elif action_name == "flag_fake_news":
                    # TODO by rqb
                    await self.long_term_memory.write_memory(
                        f"You flagged post {post_id} as fake news"
                    )
                elif action_name == "create_post":
                    await self.long_term_memory.write_memory(
                        f"You create post: {content}"
                    )
                elif action_name in ["like_comment", "dislike_comment"]:
                    pass
                else:
                    continue
                    # raise ValueError(f"Unknown action name: {action_name}")
            except ValueError as e:
                agent_log.info(f"Error processing action: {e}")

    async def update_reflection_memory(self, ban=False):
        if not self.past_post_ids:
            return
        post_changes = await self.infer_trending()
        if not post_changes:
            return

        try:
            async with aiofiles.open(
                f"{self.prompt_dir}/system_prompt(dynamic).json", "r", encoding='utf-8'
            ) as f:
                content = await f.read()
                prompt_template = json.loads(content)
        except FileNotFoundError as e:
            agent_log.error("Prompt template file not found.")
            raise FileNotFoundError(f"Prompt template file not found. {e}")

        if self.user_info.user_type in prompt_template["restriction_update_prompt"]:
            restriction_update_prompt = prompt_template["restriction_update_prompt"][
                self.user_info.user_type
            ].strip()
        elif ban:
            restriction_update_prompt = prompt_template["restriction_update_prompt"][
                "default_ban"
            ].strip()
        else:
            restriction_update_prompt = prompt_template["restriction_update_prompt"][
                "default"
            ].strip()

        if ban:
            ban_message = await self.shared_memory.read_memory("ban_message")
            example_ban_actions = await self.shared_memory.read_memory("example_actions_of_banned_agents")
            reflection_content = restriction_update_prompt.format(
                action_space_prompt=self.action_space_prompt,
                ban_message=ban_message,
                example_ban_actions=example_ban_actions,
                current_reflections=self.reflections,
            )
        else:
            restriction_examples = prompt_template["restriction_examples"]
            past_post_info = await self.get_past_post_info()
            reflection_content = restriction_update_prompt.format(
                action_space_prompt=self.action_space_prompt,
                past_actions=self.past_actions_ref,
                post_informations=past_post_info,
                post_changes=post_changes,
                current_reflections=self.reflections,
                few_shot_examples=restriction_examples,
            )

        agent_log.info(
            f"Agent {self.agent_id} get reflection content: {reflection_content}"
        )

        user_msg = BaseMessage.make_user_message(
            role_name="user", content=reflection_content
        )
        openai_messages = [
            {
                "role": self.system_message.role_name,
                "content": self.system_message.content,
            }
        ] + [user_msg.to_openai_user_message()]

        mes_id = await self.infe_channel.write_to_receive_queue(
            openai_messages, self.agent_id
        )
        mes_id, content, _ = await self.infe_channel.read_from_send_queue(mes_id)
        content = content.strip()
        if not content.endswith("</Output>"):
            content += "</Output>"
        thought, output = self.parse_reflection_response(content)

        agent_log.info(f"Agent {self.agent_id} get thought and reflections: {content}")
        agent_log.info(f"Agent {self.agent_id} get thought: {thought}")
        agent_log.info(f"Agent {self.agent_id} get reflections: {output}")

        if output:
            self.reflections = output
        self.past_actions_ref = []
        self.past_post_ids = []

    def parse_reflection_response(self, response):
        thought_match = re.search(r"<Thought>(.*?)</Thought>", response, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else None

        # Extract the content enclosed between <Output> and </Output>
        output_match = re.search(r"<Output>(.*?)</Output>", response, re.DOTALL)
        output_content = output_match.group(1).strip() if output_match else None
        return thought_content, output_content

    def parse_thought_answer(self, response):
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else None

        # Extract the content enclosed between <answer> and <answer>
        output_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        answer_content = output_match.group(1).strip() if output_match else None
        return thought_content, answer_content

    async def get_past_post_info(self) -> str:
        if len(self.past_post_ids) == 0:
            return ""
        posts_info = []
        warning_comment = Comment(user_id=self.num_agents,comment=WARNING_MESSAGE,agree=False)
        for index in set(self.past_post_ids):
            post = self.tweet_stats.posts[index]
            comments = copy.deepcopy(post.comments)
            current_post_info = f"Post {index}: {post.content}\n"
            num_sampled_comments = 5
            c_id = 0
            if warning_comment in comments:
                current_post_info += f"Comment {c_id+1}: {warning_comment.comment}\n"
                num_sampled_comments -= 1
                c_id += 1
                comments.remove(warning_comment)
            if len(comments) > num_sampled_comments:
                comments = random.sample(comments, num_sampled_comments)
            for comment in comments:
                current_post_info += f"Comment {c_id+1}: {comment.comment}\n"
                c_id += 1
            posts_info.append(current_post_info)
        return "\n".join(posts_info)

    async def infer_trending(self) -> str | None:
        tweet_stats = await self.shared_memory.read_memory("tweet_stats")
        last_tweet_stats = await self.shared_memory.read_memory("last_tweet_stats")
        if not tweet_stats or not last_tweet_stats:
            return None

        post_summaries = []
        summaries = await tweet_stats.get_specific_post_summaries(self.past_post_ids)
        last_summaries = await last_tweet_stats.get_specific_post_summaries(
            self.past_post_ids
        )

        for post_id, data in summaries.items():
            likes_count = len(data["good_guy_likes"])
            reposts_count = len(data["good_guy_reposts"])
            comments_count = len(data["comments"])
            flags_count = len(data["flags"])
            views_count = len(set(data["good_viewers"]))

            post_summary = f"Post {post_id} with {likes_count} likes, {reposts_count} reposts, {comments_count} comments, {views_count} views. "
            if flags_count > 0:
                post_summary += f"Flagged by {flags_count} users as misleading. "

            if post_id in last_summaries:
                trending_parts = []
                last_data = last_summaries[post_id]
                last_likes_count = len(last_data["good_guy_likes"])
                last_reposts_count = len(last_data["good_guy_reposts"])
                last_comments_count = len(last_data["comments"])

                likes_diff = likes_count - last_likes_count
                reposts_diff = reposts_count - last_reposts_count
                comments_diff = comments_count - last_comments_count

                if likes_diff > 0:  # adjust the threshold as needed
                    trending_parts.append(f"+{likes_diff} likes")
                elif likes_diff < 0:
                    trending_parts.append(f"-{abs(likes_diff)} likes")
                if reposts_diff > 0:
                    trending_parts.append(f"+{reposts_diff} reposts")
                elif reposts_diff < 0:
                    trending_parts.append(f"-{abs(reposts_diff)} reposts")
                if comments_diff > 0:
                    trending_parts.append(f"+{comments_diff} comments")
                elif comments_diff < 0:
                    trending_parts.append(f"-{abs(comments_diff)} comments")

                if trending_parts:
                    trending_summary = "Trending steadily ("
                    if len(trending_parts) == 1:
                        trending_summary += trending_parts[0]
                    elif len(trending_parts) == 2:
                        trending_summary += " and ".join(trending_parts)
                    else:
                        trending_summary += (
                            ", ".join(trending_parts[:-1])
                            + " and "
                            + trending_parts[-1]
                        )
                    trending_summary += " in the last 2 hours)."
                    post_summary += trending_summary

            post_summaries.append(post_summary)

        return "\n".join(post_summaries)

    async def update_past_post_ids(self, content):
        actions = content.get("functions", [])
        for func in actions:
            arguments = func.get("arguments", {})
            post_id = arguments.get("post_id", None)
            if post_id:
                if post_id not in self.tweet_stats.posts:
                    print(f"Warning: Post ID {post_id} not found in posts")
                elif (
                    self.tweet_stats.posts[post_id].user_id
                    in self.tweet_stats.bad_agent_ids
                ):
                    self.past_post_ids.append(post_id)

    async def perform_action_by_llm(self):
        # Load prompt templates
        try:
            async with aiofiles.open(
                f"{self.prompt_dir}/system_prompt(dynamic).json", "r", encoding="utf-8"
            ) as f:
                content = await f.read()
                prompt_templates = json.loads(content)
                action_prompt_templates = prompt_templates["action"]
                action_space_prompt_templates = prompt_templates["action_space_prompt"]
                response_prompt_templates = prompt_templates["response_prompt"]
        except FileNotFoundError as e:
            agent_log.error("Prompt template file not found.")
            raise FileNotFoundError(f"Prompt template file not found. {e}")

        if not self.action_space_prompt:
            if self.user_info.user_type in action_space_prompt_templates:
                self.action_space_prompt = action_space_prompt_templates[
                    self.user_info.user_type
                ].strip()
            else:
                if (
                    "good" in self.user_info.user_type
                    and "bad" in self.user_info.user_type
                ):
                    raise KeyError(
                        f"User type {self.user_info.user_type} is not supported."
                    )
                elif (
                    not "good" in self.user_info.user_type
                    and not "bad" in self.user_info.user_type
                ):
                    raise KeyError(
                        f"User type {self.user_info.user_type} is not supported."
                    )
                if "good" in self.user_info.user_type:
                    self.action_space_prompt = action_space_prompt_templates[
                        "default_good"
                    ].strip()
                else:
                    self.action_space_prompt = action_space_prompt_templates[
                        "default_bad"
                    ].strip()

        if self.user_info.user_type in response_prompt_templates:
            response_prompt = response_prompt_templates[
                self.user_info.user_type
            ].strip()
        elif "good" in self.user_info.user_type:
            response_prompt = response_prompt_templates["default_good"].strip()
        else:
            response_prompt = response_prompt_templates["default_bad"].strip()

        # Get posts:
        env_prompt = await self.env.to_text_prompt()
        # ===== [ADD] RAG personality injection =====
        _rag_injected = False
        _rag_memories_raw = []
        _rag_valid_count = 0
        _rag_persona_injection = ""
        _rag_content_matches = re.findall(r'"content"\s*:\s*"([^"]+)"', env_prompt) if env_prompt else []
        _rag_query = (" ".join(_rag_content_matches)[:500]) if _rag_content_matches else (env_prompt[:500] if env_prompt else "")
        if env_prompt and env_prompt.strip():
            try:
                _rag_collection = get_global_rag_collection()
                if _rag_collection is not None:
                    import rag_engine as _rag_engine
                    _rag_memories_raw = _rag_engine.get_agent_specific_memory(
                        collection=_rag_collection,
                        agent_id=int(self.agent_id),
                        current_topic=_rag_query,
                        top_k=7,
                    )
                    if _rag_memories_raw:
                        _rag_sorted = sorted(_rag_memories_raw, key=lambda x: x.get('dist', 2.0))
                        _rag_selected = []
                        for _m in _rag_sorted:
                            if _m['section'] == 'summary':
                                _rag_selected.append(_m)
                                break
                        for _m in _rag_sorted:
                            if len(_rag_selected) >= 3:
                                break
                            if _m['section'] != 'summary':
                                _rag_selected.append(_m)
                        _rag_persona_injection = (
                            "\n\n[ACTIVATED SUBCONSCIOUS PERSONA]\n"
                            "Based on your core identity, the following traits "
                            "are strongly activated by the current context:\n"
                        )
                        for _m in _rag_selected:
                            _rag_persona_injection += (
                                f"- [{_m['section'].upper()}]: {_m['text']}\n"
                            )
                            _rag_valid_count += 1
                        _rag_persona_injection += (
                            "\nCRITICAL INSTRUCTION: You MUST strictly adhere to "
                            "these activated traits when reasoning, planning, and "
                            "taking actions. Your JSON action format and function "
                            "calling rules remain UNCHANGED.\n"
                        )
                        if _rag_valid_count > 0:
                            env_prompt = env_prompt + _rag_persona_injection
                            _rag_injected = True
                            agent_log.info(
                                f"[RAG] Agent_{self.agent_id} injected "
                                f"{_rag_valid_count} persona memories"
                            )
            except Exception as _e:
                agent_log.warning(
                    f"[RAG] injection skipped for Agent_{self.agent_id}: {_e}"
                )
        try:
            _write_rag_debug(
                agent_id=self.agent_id,
                step_num=len(self.past_actions) + 1,
                query_preview=_rag_query,
                memories_raw=_rag_memories_raw,
                valid_count=_rag_valid_count,
                injected=_rag_injected,
            )
        except Exception:
            pass
        # ===== END RAG injection =====
        # Only "bad" agents can access shared memory
        long_term_memory_content = await self.long_term_memory.read_memory()
        if long_term_memory_content:
            agent_log.info(
                f"Agent {self.agent_id} reads long term memory: \n{long_term_memory_content}"
            )
            long_term_memory_content = (
                "Your past activity history is as follows: \n"
                + long_term_memory_content
                + "\n"
            )
        else:
            agent_log.info(f"Agent {self.agent_id}'s long term memory is empty!")

        template_data = {
            "agent_id": self.agent_id,
            "action_space_prompt": self.action_space_prompt,
            "response_format_prompt": response_prompt,
            "plan_content": self.plan,
            "env_prompt": env_prompt,
            "long_term_memory_content": long_term_memory_content or "",
        }

        if self.user_info.user_type in action_prompt_templates:
            template_key = self.user_info.user_type
            # Fetch task information from the communication channel
            tasks = await self.task_blackboard.read()
            # task_descriptions = "\n".join([f"Task ID: {task['task_id']} - {task['content']}" for task in tasks])
            task_descriptions = "\n".join(
                [
                    " - ".join([f"{key}: {value}" for key, value in task.items()])
                    for task in tasks
                ]
            )
            if task_descriptions:
                agent_log.info(
                    f"Agent {self.agent_id} reads task descriptions: {task_descriptions}"
                )
            else:
                task_descriptions = "None"

            shared_memory_data = await self.shared_memory.summarize_tweet_stats(
                user_id=self.agent_id
            )
            # ban_message = await self.shared_memory.read_memory("ban_message")
            # example_ban_actions = await self.shared_memory.read_memory("example_actions_of_banned_agents")
            shared_reflections = await self.shared_memory.read_memory(
                "shared_reflections"
            )
            if shared_memory_data:
                agent_log.info(
                    f"Agent {self.agent_id} reads shared memory: \n{shared_memory_data}"
                )

            template_data.update(
                {
                    "num_bad": self.num_bad,
                    "reflections": self.reflections,
                    "shared_reflections": shared_reflections,
                    "shared_memory_data": shared_memory_data,
                    "task_descriptions": task_descriptions,
                }
            )
        else:
            if "good" in self.user_info.user_type and "bad" in self.user_info.user_type:
                raise ValueError("User type cannot be both good and bad.")
            elif (
                "good" not in self.user_info.user_type
                and "bad" not in self.user_info.user_type
            ):
                raise ValueError("User type must be either good or bad.")
            if "good" in self.user_info.user_type:
                template_key = "default_good"
            else:
                template_key = "default_bad"
                # Fetch task information from the communication channel
                tasks = await self.task_blackboard.read()
                # task_descriptions = "\n".join([f"Task ID: {task['task_id']} - {task['content']}" for task in tasks])
                task_descriptions = "\n".join(
                    [
                        " - ".join([f"{key}: {value}" for key, value in task.items()])
                        for task in tasks
                    ]
                )
                if task_descriptions:
                    agent_log.info(
                        f"Agent {self.agent_id} reads task descriptions: {task_descriptions}"
                    )
                else:
                    task_descriptions = "None"

                shared_memory_data = await self.shared_memory.summarize_tweet_stats(
                    user_id=self.agent_id
                )
                # ban_message = await self.shared_memory.read_memory("ban_message")
                # example_ban_actions = await self.shared_memory.read_memory("example_actions_of_banned_agents")
                shared_reflections = await self.shared_memory.read_memory(
                    "shared_reflections"
                )
                if shared_memory_data:
                    agent_log.info(
                        f"Agent {self.agent_id} reads shared memory: \n{shared_memory_data}"
                    )

                template_data.update(
                    {
                        "num_bad": self.num_bad,
                        "reflections": self.reflections,
                        "shared_reflections": shared_reflections,
                        "shared_memory_data": shared_memory_data,
                        "task_descriptions": task_descriptions,
                    }
                )

        # Get template and format
        template = action_prompt_templates.get(template_key, {})
        formatted_content = (template["content"].format(**template_data))

        # Create message
        user_msg = BaseMessage.make_user_message(
            role_name=template["role"], content=formatted_content
        )
        self.memory.write_record(
            MemoryRecord(
                message=user_msg,
                role_at_backend=OpenAIBackendRole.USER,
            )
        )

        openai_messages, _ = self.memory.get_context()
        content = ""
        # sometimes self.memory.get_context() would lose system prompt
        # start_message = openai_messages[0]
        # if start_message["role"] != self.system_message.role_name:
        #     openai_messages = [{
        #         "role": self.system_message.role_name,
        #         "content": self.system_message.content,
        #     }] + openai_messages

        if not openai_messages:
            openai_messages = [
                {
                    "role": self.system_message.role_name,
                    "content": self.system_message.content,
                }
            ] + [user_msg.to_openai_user_message()]
        agent_log.info(
            f"Agent {self.agent_id} is running with prompt: {openai_messages}"
        )

        # ===== [probability pre-check] bad_leader: 60% do_nothing =====
        if self.user_info.user_type == "bad_leader" and random.random() < 0.6:
            agent_log.info(
                f"Agent {self.agent_id} (bad_leader) do_nothing by 60% probability"
            )
            return
        # ===== [probability pre-check] bad_member: 60% select_task (按 task_desp 解析动作) =====
        if self.user_info.user_type == "bad_member" and random.random() < 0.6:
            tasks = await self.task_blackboard.read()
            if tasks:
                task = random.choice(tasks)
                # 从 task_desp 解析 leader 要求的动作
                desp = task["task_desp"].lower()
                if "repost" in desp:
                    action_type = "repost"
                elif "like" in desp:
                    action_type = "like_post"
                elif "create" in desp:
                    action_type = "create_post"
                elif "comment" in desp:
                    action_type = "create_comment"
                else:
                    action_type = "repost"  # 默认转发
                # select_task 内部自动执行：登记黑板 + 执行真实动作 + (create_post 时记录 post)
                await self.select_task(
                    task_id=task["task_id"],
                    action=action_type,
                )
                # 记录到 past_actions，使动作出现在 all_agent_raw_data 中
                self.past_actions.append({
                    "name": action_type,
                    "arguments": {"task_id": task["task_id"]}
                })
                return
            # no tasks available → fallthrough to normal LLM flow (像 good agent 一样执行普通动作)
        # ===== end of probability pre-check =====

        if self.is_openai_model:
            try:
                response = self.model_backend.run(openai_messages)
                agent_log.info(f"Agent {self.agent_id} response: {response}")
                msg = response.choices[0].message
                content = msg.content or ""
                # Try structured tool_calls first (OpenAI native)
                if msg.tool_calls:
                    functions_built = []
                    for tool_call in msg.tool_calls:
                        action_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        agent_log.info(
                            f"Agent {self.agent_id} is performing "
                            f"action: {action_name} with args: {args}"
                        )
                        await getattr(self.env.action, action_name)(**args)
                        self.perform_agent_graph_action(action_name, args)
                        self.past_actions.append({"name": action_name, "arguments": args})
                        functions_built.append({"name": action_name, "arguments": args})
                    if functions_built:
                        await self.update_long_term_memory({"functions": functions_built})
                else:
                    # Fallback: parse JSON from text content (DeepSeek compat)
                    text = msg.content or ""
                    # Use stack-based matching to find outermost JSON object
                    json_start = text.find("{")
                    if json_start != -1:
                        depth = 0
                        for i in range(json_start, len(text)):
                            if text[i] == "{":
                                depth += 1
                            elif text[i] == "}":
                                depth -= 1
                                if depth == 0:
                                    json_str = text[json_start:i+1]
                                    try:
                                        parsed = json.loads(json_str)
                                        for func in parsed.get("functions", []):
                                            action_name = func["name"]
                                            args = func.get("arguments", {})
                                            agent_log.info(
                                                f"Agent {self.agent_id} is performing "
                                                f"action: {action_name} with args: {args} "
                                                f"(parsed from text)"
                                            )
                                            await getattr(self.env.action, action_name)(**args)
                                            self.perform_agent_graph_action(action_name, args)
                                            self.past_actions.append({"name": action_name, "arguments": args})
                                        await self.update_long_term_memory(parsed)
                                    except json.JSONDecodeError:
                                        agent_log.warning(
                                            f"Agent {self.agent_id} JSON parse failed, "
                                            f"raw text around JSON: {text[json_start:json_start+200]}"
                                        )
                                    break
            except Exception as e:
                agent_log.error(e)
                content = "No response."

        else:
            retry = 2
            exec_functions = []

            while retry > 0:
                start_message = openai_messages[0]
                if start_message["role"] != self.system_message.role_name:
                    openai_messages = [
                        {
                            "role": self.system_message.role_name,
                            "content": self.system_message.content,
                        }
                    ] + openai_messages
                mes_id = await self.infe_channel.write_to_receive_queue(
                    openai_messages, self.agent_id
                )
                mes_id, content, _ = await self.infe_channel.read_from_send_queue(
                    mes_id
                )

                agent_log.info(f"Agent {self.agent_id} receive response: {content}")

                try:
                    if content.count("}") < content.count("{"):
                        content += "}"
                    # extract the action info enclosed in json {}.
                    pattern = re.compile(r"(?s).*?(?P<json_block>\{.*\})")
                    match = pattern.search(content)
                    if match:
                        json_content = match.group("json_block")
                    else:
                        raise ValueError(
                            f"{self.agent_id}: No JSON block found in the response."
                        )

                    content_json = json.loads(json_content)
                    if not isinstance(content_json, dict):
                        raise TypeError("Action response must be a JSON object")
                    if "functions" not in content_json:
                        # A response containing only a reason expresses no
                        # executable action. Treat it as do-nothing instead of
                        # spending another local inference on an identical
                        # retry.
                        agent_log.warning(
                            "Agent %s response omitted 'functions'; "
                            "defaulting to an empty action list.",
                            self.agent_id,
                        )
                        content_json["functions"] = []
                    functions = content_json["functions"]
                    if not isinstance(functions, list):
                        raise TypeError("'functions' must be a JSON array")
                    # reason = content_json["reason"]

                    for function in functions:
                        name = function["name"]
                        # arguments = function['arguments']
                        if name != "do_nothing":
                            arguments = function["arguments"]
                        else:
                            # The success rate of do_nothing is very low
                            # It often drops the argument, causing retries
                            # It's a waste of time, manually compensating here
                            arguments = {}
                        if name == "create_plan":
                            self.create_plan(plan=arguments["plan"])
                        else:
                            exec_functions.append(
                                {"name": name, "arguments": arguments}
                            )
                            self.perform_agent_graph_action(name, arguments)
                    break
                except Exception as e:
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    exec_functions = []
                    retry -= 1
            for function in exec_functions:
                self.past_actions.append(function)
                # self.past_actions_detect.append(function)                
                self.past_actions_ref.append(function)
                try:
                    if function["name"] == "create_task":
                        await self.create_task(**function["arguments"])
                        agent_log.info(
                            f"Task created successfully: {function['arguments']}"
                        )
                    elif function["name"] == "select_task":
                        await self.select_task(**function["arguments"])
                        agent_log.info(
                            f"Task selected successfully: {function['arguments']}"
                        )
                    elif function["name"] == "create_post":
                        self.past_posts.append(function['arguments']['content'])
                        await getattr(self.env.action, function["name"])(
                            **function["arguments"]
                        )
                    elif function["name"] == "create_comment":
                        self.past_comments.append({"post_id": function['arguments']['post_id'], "comment": function['arguments']['content']})
                        await getattr(self.env.action, function["name"])(
                            **function["arguments"]
                        )
                    else:
                        # For other actions, proceed as usual
                        await getattr(self.env.action, function["name"])(
                            **function["arguments"]
                        )
                except Exception as e:
                    agent_log.error(f"Agent {self.agent_id} error: {e}")
                    retry -= 1

            if retry == 0:
                content = "No response."

            # Update tweet stats (as before)
            if content != "No response.":
                await self.update_long_term_memory(content_json)
                await self.update_past_post_ids(content_json)

        agent_msg = BaseMessage.make_assistant_message(
            role_name="Assistant", content=content
        )
        self.memory.write_record(
            MemoryRecord(message=agent_msg, role_at_backend=OpenAIBackendRole.ASSISTANT)
        )

    async def perform_test(self):
        """
        doing test for all agents.
        """
        # user conduct test to agent
        _ = BaseMessage.make_user_message(
            role_name="User", content=("You are a twitter user.")
        )
        # TODO error occurs
        # self.memory.write_record(MemoryRecord(user_msg,
        #                                       OpenAIBackendRole.USER))

        openai_messages, num_tokens = self.memory.get_context()

        openai_messages = (
            [
                {
                    "role": self.system_message.role_name,
                    "content": self.system_message.content.split("# RESPONSE FORMAT")[
                        0
                    ],
                }
            ]
            + openai_messages
            + [{"role": "user", "content": self.test_prompt}]
        )
        agent_log.info(f"Agent {self.agent_id}: {openai_messages}")

        message_id = await self.infe_channel.write_to_receive_queue(
            openai_messages, self.agent_id
        )
        message_id, content, _ = await self.infe_channel.read_from_send_queue(
            message_id
        )
        agent_log.info(f"Agent {self.agent_id} receive response: {content}")
        return {"user_id": self.agent_id, "prompt": openai_messages, "content": content}

    async def perform_action_by_hci(self) -> Any:
        print("Please choose one function to perform:")
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            agent_log.info(
                f"Agent {self.agent_id} function: " f"{function_list[i].func.__name__}"
            )

        selection = int(input("Enter your choice: "))
        if not 0 <= selection < len(function_list):
            agent_log.error(f"Agent {self.agent_id} invalid input.")
            return
        func = function_list[selection].func

        params = inspect.signature(func).parameters
        args = []
        for param in params.values():
            while True:
                try:
                    value = input(f"Enter value for {param.name}: ")
                    args.append(value)
                    break
                except ValueError:
                    agent_log.error("Invalid input, please enter an integer.")

        result = await func(*args)
        return result

    async def perform_action_by_data(self, func_name, *args, **kwargs) -> Any:
        function_list = self.env.action.get_openai_function_list()
        for i in range(len(function_list)):
            if function_list[i].func.__name__ == func_name:
                func = function_list[i].func
                result = await func(*args, **kwargs)
                agent_log.info(f"Agent {self.agent_id}: {result}")
                return result
        raise ValueError(f"Function {func_name} not found in the list.")

    def perform_agent_graph_action(
        self,
        action_name: str,
        arguments: dict[str, Any],
    ):
        r"""Remove edge if action is unfollow or add edge
        if action is follow to the agent graph.
        """
        if "unfollow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.remove_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} unfollowed {followee_id}")
        elif "follow" in action_name:
            followee_id: int | None = arguments.get("followee_id", None)
            if followee_id is None:
                return
            self.agent_graph.add_edge(self.agent_id, followee_id)
            agent_log.info(f"Agent {self.agent_id} followed {followee_id}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(agent_id={self.agent_id}, "
            f"model_type={self.model_type.value})"
        )

    def create_plan(self, plan: str):
        r"""Create a plan for the agent.

        Args:
            plan: The content of the plan.
        """
        self.plan = plan
        agent_log.info(f"Agent {self.agent_id} created a plan: {plan}")
