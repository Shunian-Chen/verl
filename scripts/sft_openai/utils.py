from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

class QuestionStrategy(Enum):
    """Question generation strategies"""
    VISUAL_PERCEPTION = "visual_perception"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    INFERENTIAL = "inferential"
    META_COGNITIVE = "meta_cognitive"


@dataclass
class GeneratedExample:
    """Represents a generated training example"""
    id: str
    image: str
    wiki_title: str
    categories: List[str]
    question: str
    question_strategy: str
    complexity: str
    response: str
    num_cycles: int
    word_count: int
    gpt_generation_metadata: Dict[str, Any]
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    is_multiple_choice: bool = False
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class APIUsageStats:
    """Track API usage and costs"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    generation_requests: int = 0
    validation_requests: int = 0
    retry_count: int = 0

    def add_usage(self, prompt_tokens: int, completion_tokens: int,
                  model: str, request_type: str = "generation"):
        """Add usage statistics"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

        # Cost calculation (as of Nov 2025)
        # GPT-4 Turbo: $10/1M input, $30/1M output
        # GPT-3.5 Turbo: $0.5/1M input, $1.5/1M output
        if "gpt-4" in model.lower():
            input_cost = prompt_tokens * 10 / 1_000_000
            output_cost = completion_tokens * 30 / 1_000_000
        else:  # GPT-3.5 Turbo
            input_cost = prompt_tokens * 0.5 / 1_000_000
            output_cost = completion_tokens * 1.5 / 1_000_000

        self.total_cost_usd += input_cost + output_cost

        if request_type == "generation":
            self.generation_requests += 1
        elif request_type == "validation":
            self.validation_requests += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)