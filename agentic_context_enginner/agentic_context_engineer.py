from typing import List, Dict, Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
from crewai import Agent, Task, Crew, Process, LLM
import json
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================
# Data Models
# ============================================
class DeltaOperation(BaseModel):
    """Single mutation to apply to the playbook."""
    model_config = ConfigDict(extra="ignore")
    
    type: Literal["ADD", "UPDATE", "REMOVE"]
    section: str = Field(default="general")
    content: Optional[str] = None
    bullet_id: Optional[str] = None


class DeltaBatch(BaseModel):
    """Bundle of curator reasoning and delta operations."""
    model_config = ConfigDict(extra="ignore")
    
    reasoning: str
    operations: List[DeltaOperation] = Field(default_factory=list)


class Bullet(BaseModel):
    """Single playbook entry."""
    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    def tag(self, tag: Literal["helpful", "harmful", "neutral"], increment: int = 1):
        current = getattr(self, tag)
        setattr(self, tag, current + increment)
        self.updated_at = datetime.now(timezone.utc).isoformat()


class Playbook(BaseModel):
    """Structured context store for self-improving knowledge."""
    bullets: Dict[str, Bullet] = Field(default_factory=dict)
    sections: Dict[str, List[str]] = Field(default_factory=dict)
    next_id: int = 0
    
    def add_bullet(self, section: str, content: str, bullet_id: Optional[str] = None) -> Bullet:
        bullet_id = bullet_id or self._generate_id(section)
        bullet = Bullet(id=bullet_id, section=section, content=content)
        self.bullets[bullet_id] = bullet
        self.sections.setdefault(section, []).append(bullet_id)
        return bullet
    
    def update_bullet(self, bullet_id: str, content: str) -> Optional[Bullet]:
        bullet = self.bullets.get(bullet_id)
        if bullet is None:
            return None
        bullet.content = content
        bullet.updated_at = datetime.now(timezone.utc).isoformat()
        return bullet
    
    def remove_bullet(self, bullet_id: str):
        bullet = self.bullets.pop(bullet_id, None)
        if bullet is None:
            return
        section_list = self.sections.get(bullet.section)
        if section_list:
            self.sections[bullet.section] = [bid for bid in section_list if bid != bullet_id]
            if not self.sections[bullet.section]:
                del self.sections[bullet.section]
    
    def update_bullet_tag(self, bullet_id: str, tag: Literal["helpful", "harmful", "neutral"], 
                        increment: int = 1) -> Optional[Bullet]:
        bullet = self.bullets.get(bullet_id)
        if bullet:
            bullet.tag(tag, increment)
        return bullet
    
    def apply_delta(self, delta: DeltaBatch):
        for operation in delta.operations:
            if operation.type == "ADD":
                self.add_bullet(operation.section, operation.content, operation.bullet_id)
            elif operation.type == "UPDATE" and operation.bullet_id:
                self.update_bullet(operation.bullet_id, operation.content)
            elif operation.type == "REMOVE" and operation.bullet_id:
                self.remove_bullet(operation.bullet_id)
    
    def as_prompt(self) -> str:
        """Return human-readable playbook for prompting."""
        parts = []
        for section, bullet_ids in sorted(self.sections.items()):
            parts.append(f"## {section}")
            for bullet_id in bullet_ids:
                bullet = self.bullets[bullet_id]
                counters = f"(helpful={bullet.helpful}, harmful={bullet.harmful}, neutral={bullet.neutral})"
                parts.append(f"- [{bullet.id}] {bullet.content} {counters}")
        return "\n".join(parts) if parts else "No playbook entries yet."
    
    def _generate_id(self, section: str) -> str:
        self.next_id += 1
        section_prefix = (section or "general").split()[0].lower()
        return f"{section_prefix}-{self.next_id:05d}"
    
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str):
        """Load playbook from JSON file, handling missing or empty files."""
        if not os.path.exists(path):
            return cls()
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:  # Handle empty file
                    return cls()
                return cls(**json.loads(content))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {path}. Creating new playbook. Error: {e}")
            return cls()
        except Exception as e:
            print(f"Warning: Error loading {path}. Creating new playbook. Error: {e}")
            return cls()


class GeneratorOutput(BaseModel):
    reasoning: List[str] = Field(description="Step-by-step reasoning process")
    bullet_ids: List[str] = Field(default_factory=list, description="Playbook bullets referenced")
    final_answer: str = Field(description="Concise final answer")


class BulletTag(BaseModel):
    id: str
    tag: Literal["helpful", "harmful", "neutral"]


class Reflection(BaseModel):
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag] = Field(default_factory=list)


# ============================================
# Agentic RAG Crew
# ============================================
class AgenticCrew:
    def __init__(self, model: str, api_key: str, base_url: str = None,playbook_path: str = "playbook.json"):
        """
        Initialize Agentic RAG Crew.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini", "ollama/llama3.2", "groq/llama-3.1-70b-versatile")
            api_key: API key for the model provider
            base_url: Optional base URL (for Ollama or custom endpoints)
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.playbook_path = playbook_path
        self.playbook = Playbook.load(playbook_path)
        self.context = {}
    
    def create_generator_agent(self) -> Agent:
        """Agent that generates answers using the playbook."""
        return Agent(
            role="Answer Generator",
            goal="Solve problems by referencing the playbook and providing structured reasoning",
            backstory="""You are an expert problem solver who carefully references a curated 
            playbook of strategies and insights. You show your reasoning step-by-step and 
            track which playbook entries you used.""",
            verbose=True,
            allow_delegation=False,
            llm=LLM(model=self.model, api_key=self.api_key, base_url=self.base_url,temperature=0) if self.base_url 
                else LLM(model=self.model, api_key=self.api_key,temperature=0)
        )
    
    def create_reflector_agent(self) -> Agent:
        """Agent that critically analyzes reasoning and identifies improvements."""
        return Agent(
            role="Critical Reflector",
            goal="Analyze reasoning traces to identify errors, patterns, and improvement opportunities",
            backstory="""You are a meticulous analyst who examines problem-solving approaches 
            to find what went wrong and why. You identify root causes and provide actionable 
            insights for improvement.""",
            verbose=True,
            allow_delegation=False,
            llm=LLM(model=self.model, api_key=self.api_key, base_url=self.base_url,temperature=0) if self.base_url 
                else LLM(model=self.model, api_key=self.api_key,temperature=0)
        )
    
    def create_curator_agent(self) -> Agent:
        """Agent that curates and updates the playbook."""
        return Agent(
            role="Playbook Curator",
            goal="Maintain a high-quality playbook by adding new insights and removing outdated content",
            backstory="""You are an expert curator who distills reflections into actionable 
            playbook entries. You avoid duplication, focus on quality over quantity, and 
            ensure the playbook remains organized and useful.""",
            verbose=True,
            allow_delegation=False,
            llm=LLM(model=self.model, api_key=self.api_key, base_url=self.base_url,temperature=0) if self.base_url 
                else LLM(model=self.model, api_key=self.api_key,temperature=0)
        )
    
    def create_generator_task(self, agent: Agent, user_query: str) -> Task:
        """Task for generating an answer."""
        playbook_text = self.playbook.as_prompt()
        
        return Task(
            description=f"""Your task is to answer user queries while providing structured step-by-step reasoning and the bullet IDs you used.
                        input:
                            -User Query: {user_query}
                            -Current Playbook:{playbook_text}

【Required Guidelines】

1. Carefully read the playbook and apply relevant strategies, formulas, and insights
- Check all bullet points in the playbook
- Understand the context and application conditions of each strategy

2. Carefully examine common failures (anti-patterns) listed in the playbook and avoid them
- Present specific alternatives or best practices

3. Show the reasoning process step by step
- Clearly indicate which bullets you referenced at each stage
- Structure so that the logic flow is clear

4. Create thorough but concise analysis
- Include only essential information, but include all central evidence
- Avoid unnecessary repetition

5. Review calculations and logic before providing the final answer
- Confirm that all referenced bullet_ids were actually used
- Check for logical contradictions
- Double-check that you haven't missed any relevant playbook bullets

【Output Rules】
- reasoning: Step-by-step thought process (step-by-step chain of thought), detailed analysis and calculations
- bullet_ids: List of referenced playbook bullet IDs
- final_answer: Clear and verified final answer
""",
            agent=agent,
            expected_output="JSON with reasoning steps, bullet IDs used, and final answer"
        )
    
    def create_reflector_task(self, agent: Agent, user_query: str, generator_output: str) -> Task:
        """Task for reflecting on the generation."""
        playbook_text = self.playbook.as_prompt()
        
        return Task(
            description=f"""Your task is to carefully examine the generator's output, critically analyze it, and create a reflection (JSON).
                        input:
                            -User Query: {user_query}
                            -Generator Output: {generator_output}
                            -Playbook Context:{playbook_text}


【Required Analysis Steps】

1. Carefully analyze the model's reasoning trace to understand where errors occurred
- Review the generator's entire reasoning
- Check for leaps or contradictions in the logic flow

2. Identify specific error types: conceptual errors, calculation mistakes, strategy misuse, etc.
- Clearly describe the characteristics of each error
- Find the root causes behind surface-level errors

3. Provide actionable insights so the model doesn't make the same mistakes in the future
- Present specific procedures or checklists
- Derive generalizable principles

4. Evaluate each bullet point used by the generator
- Tag each bullet_id as ['helpful', 'harmful', 'neutral']
- helpful: bullets that helped with the correct answer
- harmful: incorrect or misleading bullets that led to wrong answers
- neutral: bullets that didn't affect the final result

【Output Rules】
- reasoning: Thought process that went through all 4 analysis steps above, detailed analysis and evidence
- error_identification: Specifically describe what exactly was wrong in the reasoning
- root_cause_analysis: What was the root cause of this error? Which concepts were misunderstood? Which strategies were misused?
- correct_approach: What should the generator have done instead? Present accurate steps and logic
- key_insight: Strategy, formula, principle, or checklist that should be remembered to avoid such errors
- bullet_tags: Tagging results for each bullet referenced by the generator (including id and 'helpful'/'harmful'/'neutral')""",
            agent=agent,
            expected_output="JSON with detailed reflection and bullet tags"
        )
    
    def create_curator_task(self, agent: Agent, user_query: str, reflection: str) -> Task:
        """Task for curating the playbook."""
        playbook_text = self.playbook.as_prompt()
        
        return Task(
            description=f"""You are an expert in curating playbooks.

Considering the existing playbook and reflections from previous attempts:
- Identify only new insights, strategies, and failures that are **missing** from the current playbook
- You can **improve existing bullets with better content** or **remove erroneous/duplicate items**
- Avoid duplication - if similar advice already exists, add only new content that perfectly complements the existing playbook
- Do not regenerate the entire playbook - provide only necessary additions/modifications/deletions
- Focus on quality over quantity - a focused and organized playbook is better than a comprehensive one
- Each change must be specific and justified

User Query: {user_query}

Reflection: {reflection}

Current Playbook:
{playbook_text}


CRITICAL: You must respond with ONLY valid JSON. No markdown, no explanations, no code blocks.

Response format (maximum 3 operations per response):
{{
"reasoning": "Brief explanation (max 200 characters)",
"operations": [
    {{
    "type": "ADD",
    "section": "general",
    "content": "Specific actionable advice (max 150 characters)"
    }},
    {{
    "type": "UPDATE", 
    "bullet_id": "existing-id",
    "content": "Improved content (max 150 characters)"
    }},
    {{
    "type": "REMOVE",
    "bullet_id": "id-to-remove"
    }}
]
}}

Rules:
- Maximum 3 operations per response
- Keep content concise and actionable
- Ensure all JSON strings are properly escaped
- If no changes needed, return: {{"reasoning": "No changes needed", "operations": []}}""",

            agent=agent,
            expected_output="JSON with reasoning and operations (ADD/UPDATE/REMOVE)"
        )
    
    
    def extract_json(self, text: str) -> dict:
        """Extract JSON from text that may contain markdown, think tags, or other noise."""
        text = str(text)
        
        # Remove <think> tags and their content
        import re
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # If all else fails, return empty structure
        return {}
    def run_ace_cycle(self, user_query: str) -> Dict:
        """Execute one complete ACE (Agent-Curator-Evaluator) cycle."""
        
        # Create agents
        generator = self.create_generator_agent()
        reflector = self.create_reflector_agent()
        curator = self.create_curator_agent()
        
        # Create and execute generator task
        gen_task = self.create_generator_task(generator, user_query)
        gen_crew = Crew(agents=[generator], tasks=[gen_task], process=Process.sequential)
        gen_result = gen_crew.kickoff()
        
        try:
            gen_output = self.extract_json(str(gen_result))
        except:
            gen_output = {"reasoning": [str(gen_result)], "bullet_ids": [], "final_answer": str(gen_result)}
        
        # Create and execute reflector task
        ref_task = self.create_reflector_task(reflector, user_query, json.dumps(gen_output))
        ref_crew = Crew(agents=[reflector], tasks=[ref_task], process=Process.sequential)
        ref_result = ref_crew.kickoff()
        
        try:
            ref_output = self.extract_json(str(ref_result))
            # Update bullet tags in playbook
            for bullet_tag in ref_output.get("bullet_tags", []):
                self.playbook.update_bullet_tag(bullet_tag["id"], bullet_tag["tag"])
        except:
            ref_output = {"reasoning": str(ref_result), "bullet_tags": []}
        
        # Create and execute curator task
        cur_task = self.create_curator_task(curator, user_query, json.dumps(ref_output))
        cur_crew = Crew(agents=[curator], tasks=[cur_task], process=Process.sequential)
        cur_result = cur_crew.kickoff()
        
        try:
            cur_output = self.extract_json(str(cur_result))
            # Apply playbook updates
            delta_batch = DeltaBatch(**cur_output)
            self.playbook.apply_delta(delta_batch)
        except Exception as e:
            print(f"Curator update failed: {e}")
            cur_output = {"reasoning": str(cur_result), "operations": []}
        
        self.playbook.save(self.playbook_path)
        
        return {
            "user_query": user_query,
            "generator_output": gen_output,
            "reflector_output": ref_output,
            "curator_output": cur_output,
            "playbook_stats": {
                "sections": len(self.playbook.sections),
                "bullets": len(self.playbook.bullets)
            }
        }


# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
        # Example 3: Using Groq
    ace_crew = AgenticCrew(
        model="groq/llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API")
    )
    
    # Run multiple cycles to build up the playbook
    # queries =["A company is designing a recommendation system for an e-commerce website. Should they prioritize accuracy of recommendations or speed/latency for user interactions? Explain the trade-offs and when each priority makes sense."]
    # print("=" * 80)
    # print("AGENTIC RAG WITH SELF-IMPROVING PLAYBOOK")
    # print("=" * 80)
    
    # for i, query in enumerate(queries, 2):
    #     print(f"\n\n{'='*80}")
    #     print(f"CYCLE {i}: {query}")
    #     print(f"{'='*80}\n")

    query=""
    result = ace_crew.run_ace_cycle(query)
    
    
    print(f"\n📊 FINAL ANSWER: {result['generator_output'].get('final_answer', 'N/A')}")
    print(f"\n📚 PLAYBOOK STATS: {result['playbook_stats']}")
    print(f"\n📝 CURATOR REASONING: {result['curator_output'].get('reasoning', 'N/A')}")
    
    print("\n\n" + "="*80)
    print("FINAL PLAYBOOK")
    print("="*80)
    print(ace_crew.playbook.as_prompt())