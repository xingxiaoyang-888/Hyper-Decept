"""
profile_chunker.py

将 deeppersonal_agents.json 中的每个 profile 拆成多个语义块，
展平为自然语言文本，供后续 embedding 和向量检索使用。

输出: chunked_profiles.json（可读，用于检视 chunk 质量）
"""

import json
import os

# 需要作为独立 chunk 的顶层 section
SECTION_MAPPING = {
    "demographic": "Demographic Information",
    "career": "Career and Work Identity",
    "values": "Core Values, Beliefs, and Philosophy",
    "lifestyle": "Lifestyle and Daily Routine",
    "social_context": "Cultural and Social Context",
    "interests": "Hobbies, Interests, and Lifestyle",
}

# profile 顶层中不需要纳入 chunk 的字段
SKIP_KEYS = {"Generated At", "Profile Index"}


def flatten_dict(d: dict, parent_key: str = "") -> list[tuple[str, str]]:
    """
    递归展平嵌套字典，返回 [(展平后key, 叶子值), ...]。
    自动过滤空值和无意义占位值。
    """
    NOISE_VALUES = {
        "none", "none;", "not applicable", "", "none mentioned",
        "not specified", "no", "not interested", "none; no",
        "none; not applicable",
    }

    items: list[tuple[str, str]] = []
    for k, v in d.items():
        full_key = f"{parent_key} → {k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, full_key))
        else:
            val_str = str(v).strip() if v is not None else ""
            if val_str.lower() not in NOISE_VALUES:
                items.append((full_key, val_str))
    return items


def build_section_text(section_name: str, section_data: dict) -> str:
    """
    将一个 section dict 展平为可读的纯文本。
    例如:
      [demographic]
      Financial Status → UserStatus: Middle-income earner with modest savings
      Financial Status → FinancialBehavior: Cautious spender
    """
    pairs = flatten_dict(section_data)
    if not pairs:
        return ""

    lines = [f"[{section_name}]"]
    for key, val in pairs:
        clean_key = key.replace("_", " ")
        lines.append(f"{clean_key}: {val}")
    return "\n".join(lines)


def chunk_profile(profile: dict, agent_id: int) -> list[dict]:
    """将一个 profile 拆成多个 chunk dict。"""
    chunks: list[dict] = []

    # 1. Summary 块（原样保留，这是最宝贵的自然语言人格描述）
    summary = profile.get("Summary", "")
    if summary and summary.strip():
        chunks.append({
            "agent_id": agent_id,
            "section": "summary",
            "text": summary.strip(),
        })

    # 2. 其余语义块
    for section_key, json_key in SECTION_MAPPING.items():
        section_data = profile.get(json_key)
        if isinstance(section_data, dict):
            text = build_section_text(section_key, section_data)
            if text.strip():
                chunks.append({
                    "agent_id": agent_id,
                    "section": section_key,
                    "text": text,
                })

    return chunks


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "deeppersonal_agents.json")
    output_path = os.path.join(script_dir, "chunked_profiles.json")

    # 读入
    with open(input_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    print(f"读取 {len(profiles)} 个 profiles")

    # 逐个 chunk
    all_chunks: list[dict] = []
    for profile in profiles:
        idx = profile.get("Profile Index", 1) - 1  # 转为 0-based agent_id
        chunks = chunk_profile(profile, idx)
        all_chunks.extend(chunks)

    # 写出
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 打印统计
    print(f"\n[OK] 拆块完成: {len(profiles)} profiles -> {len(all_chunks)} chunks")
    print(f"  输出文件: {output_path}")

    sections = sorted(set(c["section"] for c in all_chunks))
    print(f"  chunk 类型: {', '.join(sections)}")

    print("\n各 chunk 详情:")
    for c in all_chunks:
        char_count = len(c["text"])
        line_count = c["text"].count("\n") + 1
        print(f"  [agent_{c['agent_id']}] {c['section']:15s}  {char_count:5d} chars, {line_count:2d} lines")
        if char_count == 0:
            print("    ⚠️ 空 chunk，请检查源数据")


if __name__ == "__main__":
    main()
