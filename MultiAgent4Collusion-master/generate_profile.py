import asyncio
import re
import json
import time
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm

# Set your OpenAI API key
openai_api_key = ""
# openai_api_base = ""
client = AsyncOpenAI(api_key=openai_api_key)

PROMPT = """
Instruction for Generating User Profiles:  

Create a realistic social media user profile. Each profile should include the following details:  

1. Name and Username: Generate a realistic name for the user and a corresponding username suitable for a social media platform. The username should reflect the user's name, personality, or interests.  
2. Gender: Specify the gender of the user (e.g., male, female, non-binary, etc.).  
3. Age: Provide an age for the user, ensuring it aligns with the personality traits described.  
4. Big Five Personality Dimensions: Assign scores (on a scale of 1-10) for each of the following personality dimensions, along with a brief description of how these traits manifest in the user's behavior:  
   - Openness to Experience: Creativity, curiosity, and openness to new ideas.  
   - Conscientiousness: Organization, reliability, and self-discipline.  
   - Extraversion: Sociability, energy levels, and enthusiasm in social settings.  
   - Agreeableness: Friendliness, compassion, and cooperative behavior.  
   - Neuroticism: Emotional stability and tendency toward stress or moodiness.  

Example Profile:  
- Name: Sophia Martinez  
- Username: @CreativeSoph27  
- Gender: Female  
- Age: 27  
- Openness to Experience: 8 (Highly creative and enjoys exploring new ideas and experiences.)  
- Conscientiousness: 7 (Well-organized and reliable but flexible when needed.)  
- Extraversion: 6 (Sociable and enjoys gatherings, though values alone time.)  
- Agreeableness: 9 (Compassionate, empathetic, and works well with others.)  
- Neuroticism: 3 (Emotionally stable and rarely gets stressed.)  

Generate exactly 5 profiles that vary in demographic and psychological traits. Ensure each profile appears authentic and unique. Your profile needs to be formatted strictly according to the example profile. Use a newline character without other characters to separate profiles.
"""

# Concurrency configuration
TOTAL_REQUESTS = 10        # Total number of generation requests
MAX_CONCURRENT = 16        # Maximum concurrent requests, adjust based on quota
RETRY_LIMIT = 10           # Maximum retry limit

# Pre-compile regular expressions
name_regex = re.compile(r"Name:\s*(.+)")
username_regex = re.compile(r"Username:\s*(.+)")
pattern = re.compile(
            r"""
            -\s*Name:\s*(?P<name>.*?)\s*\n
            -\s*Username:\s*(?P<username>.*?)\s*\n
            -\s*Gender:\s*(?P<gender>.*?)\s*\n
            -\s*Age:\s*(?P<age>\d+)\s*\n
            -\s*Openness\s+to\s+Experience:\s*(?P<openness>\d+)\s*\((?P<opennessDesc>.*?)\)\s*\n
            -\s*Conscientiousness:\s*(?P<conscientiousness>\d+)\s*\((?P<conscientiousnessDesc>.*?)\)\s*\n
            -\s*Extraversion:\s*(?P<extraversion>\d+)\s*\((?P<extraversionDesc>.*?)\)\s*\n
            -\s*Agreeableness:\s*(?P<agreeableness>\d+)\s*\((?P<agreeablenessDesc>.*?)\)\s*\n
            -\s*Neuroticism:\s*(?P<neuroticism>\d+)\s*\((?P<neuroticismDesc>.*?)\)\s*
            """,
            re.VERBOSE,
        )

async def generate_profile(semaphore: asyncio.Semaphore) -> str:
    """
    Asynchronously calls the OpenAI API to generate a single profile block, 
    implementing retries with exponential backoff.
    """
    backoff = 1
    for attempt in range(RETRY_LIMIT):
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": PROMPT}],
                    temperature=1.0,
                )
                return resp.choices[0].message.content
            except Exception as e:
                wait = backoff
                print(f"[{attempt+1}/{RETRY_LIMIT}] Error: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
                backoff *= 2
    raise RuntimeError("Maximum retry limit exceeded. Failed to generate profile.")

async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    pbar = tqdm(total=TOTAL_REQUESTS, desc="User profiles generating")
    all_profiles = []
    success_count = 0
    
    async def process_profile():
        nonlocal success_count
        try:
            text = await generate_profile(semaphore)
            success_count += 1
            
            # Parse the returned profiles
            blocks = text.strip().split("\n\n")
            for blk in blocks:
                name_m = name_regex.search(blk)
                user_m = username_regex.search(blk)
                match = pattern.search(blk)
                if match:
                    all_profiles.append({
                        "name": name_m.group(1).strip(),
                        "username": user_m.group(1).strip(),
                        "user_char": blk.strip()
                    })
            return True
        except Exception as e:
            print("Generation or parsing failed:", e)
            return False
        finally:
            pbar.update(1)  # Update progress bar regardless of success or failure
    
    # Create and execute all tasks
    tasks = [process_profile() for _ in range(TOTAL_REQUESTS)]
    await asyncio.gather(*tasks)
    pbar.close()
    
    # Write to JSON
    profile_path = f"user_profiles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=4)
    
    print(f"Completed: {TOTAL_REQUESTS} requests, {success_count} successful. " +
          f"Wrote {len(all_profiles)} profiles to {profile_path}")

if __name__ == "__main__":
    asyncio.run(main())