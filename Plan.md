# Agentic Rubrics RL 实现计划                                                                                                                                                                                        
## 概述                                                                                                                                                                                      
基于 `framework.tex` 实现 Agentic Rubrics RL 框架：将 rubric 从静态评估器升级为**可执行的、与环境耦合的奖励程序**。                                                                       
                                                                                                                                                                                            
  **核心创新**：                                                                                                                                                                            
  - Rubric 作为 agent 和环境之间的"胶水"                                                                                                                                                    
  - 通过工具调用收集可验证的证据                                                                                                                                                            
  - 两阶段流程：离线 Rubric 合成 + 在线 RL 训练                                                                                                                                             
                                                                                                                                                                                            
  ## 架构 (Framework 四层)                                                                                                                                                                  
                                                                                                                                                                                            
  ```                                                                                                                                                                                       
  ┌─────────────────────────────────────────────────────────────────────┐                                                                                                                   
  │                    Phase 1: Offline Rubric Synthesis                │                                                                                                                   
  │                                                                     │                                                                                                                   
  │  ┌──────────────┐     Step 1: Intent        ┌─────────────────┐   │                                                                                                                     
  │  │ Dataset D    │ ─────────────────────────> │ Rubric          │   │                                                                                                                    
  │  │ {x_i, spec?} │     Extraction            │ Direction d_i   │   │                                                                                                                     
  │  └──────────────┘                           └────────┬────────┘   │                                                                                                                     
  │                                                      │             │                                                                                                                    
  │                                                      v             │                                                                                                                    
  │                           Step 2: Tool Compilation                 │                                                                                                                    
  │                    ┌─────────────────────────────────────┐         │                                                                                                                    
  │                    │ G_ψ(x_i, d_i, T_e^rub)             │         │                                                                                                                     
  │                    │    → ExecutableRubric ρ_i          │         │                                                                                                                     
  │                    │    → LatencyEstimate L_i           │         │                                                                                                                     
  │                    └──────────────────┬──────────────────┘         │                                                                                                                    
  │                                       │                            │                                                                                                                    
  │                                       v                            │                                                                                                                    
  │                    ┌─────────────────────────────────────┐         │                                                                                                                    
  │                    │ D' = {(x_i, ρ_i, L_i)}             │         │                                                                                                                     
  │                    │ (Bucketed by latency)               │         │                                                                                                                    
  │                    └─────────────────────────────────────┘         │                                                                                                                    
  └─────────────────────────────────────────────────────────────────────┘                                                                                                                   
  │                                                                                                                                                                                         
  v                                                                                                                                                                                         
  ┌─────────────────────────────────────────────────────────────────────┐                                                                                                                   
  │                    Phase 2: Online RL Training                      │                                                                                                                   
  │                                                                     │                                                                                                                   
  │  ┌──────────────────────────────────────────────────────────────┐  │                                                                                                                    
  │  │ 1. Policy Rollout                                            │  │                                                                                                                    
  │  │                                                              │  │                                                                                                                    
  │  │    ┌──────────────┐      tool calls      ┌───────────────┐  │  │                                                                                                                     
  │  │    │ Policy π_θ   │ <==================> │ T_e^pol       │  │  │                                                                                                                     
  │  │    │              │      observations    │ (可见工具)     │  │  │                                                                                                                    
  │  │    └──────┬───────┘                      └───────────────┘  │  │                                                                                                                     
  │  │           │                                                  │  │                                                                                                                    
  │  │           v (y, τ)                                          │  │                                                                                                                     
  │  └──────────────────────────────────────────────────────────────┘  │                                                                                                                    
  │                                                                     │                                                                                                                   
  │  ┌──────────────────────────────────────────────────────────────┐  │                                                                                                                    
  │  │ 2. Rubric Execution                                          │  │                                                                                                                    
  │  │                                                              │  │                                                                                                                    
  │  │    ┌──────────────────┐                ┌───────────────────┐│  │                                                                                                                     
  │  │    │ Rubric Executor  │    queries     │ T_e^rub           ││  │                                                                                                                     
  │  │    │ R(x, y, τ, ρ)    │ <============> │ (全部工具)        ││  │                                                                                                                     
  │  │    │                  │    evidence    │ T_e^pol ∪ T_e^ver ││  │                                                                                                                     
  │  │    └────────┬─────────┘                └───────────────────┘│  │                                                                                                                     
  │  │             │                                                │  │                                                                                                                    
  │  │             v                                                │  │                                                                                                                    
  │  │    ┌──────────────────────────────────────────────────────┐ │  │                                                                                                                     
  │  │    │ Evidence Records E = {e_1, ..., e_J}                 │ │  │                                                                                                                     
  │  │    │                                                      │ │  │                                                                                                                     
  │  │    │ for each check v_j in V:                             │ │  │                                                                                                                     
  │  │    │     score_j = φ_j(y, e_j)                            │ │  │                                                                                                                     
  │  │    │                                                      │ │  │                                                                                                                     
  │  │    │ R(x,y) = Agg({score_j}) with gates                   │ │  │                                                                                                                     
  │  │    └──────────────────────────────────────────────────────┘ │  │                                                                                                                     
  │  └──────────────────────────────────────────────────────────────┘  │                                                                                                                    
  │                                                                     │                                                                                                                   
  │  ┌──────────────────────────────────────────────────────────────┐  │                                                                                                                    
  │  │ 3. GRPO Optimization                                         │  │                                                                                                                    
  │  │                                                              │  │                                                                                                                    
  │  │    θ ← θ + ∇_θ J(θ)  where J(θ) = E[R(x,y)]                │  │                                                                                                                      
  │  └──────────────────────────────────────────────────────────────┘  │                                                                                                                    
  └─────────────────────────────────────────────────────────────────────┘                                                                                                                   
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  **组件映射**：                                                                                                                                                                            
  | Framework | Verl 实现 | 状态 |                                                                                                                                                          
  |-----------|----------|------|                                                                                                                                                           
  | Policy Layer $\pi_\theta$ | Actor rollout workers | ✅ |                                                                                                                                
  | Policy-Tool Interaction $\mathcal{T}_e^{pol}$ | `tool_agent_loop` + `BaseTool` | ✅ |                                                                                                   
  | Environment Layer $\mathcal{T}_e$ | `BaseTool` + MCP + SandboxFusion | ✅ |                                                                                                             
  | Rubric Generator $\mathcal{G}_\psi$ (2-stage) | `LLMRubricGenerator` | ❌ 需实现 |                                                                                                      
  | Rubric Executor $\mathcal{R}$ | `RubricExecutor` | ❌ 需实现 |                                                                                                                          
  | Time-bucketing | `LatencyBucketedSampler` | ❌ 需实现 |                                                                                                                                 
                                                                                                                                                                                            
  ## 模块结构                                                                                                                                                                               
                                                                                                                                                                                            
  ```                                                                                                                                                                                       
  verl/                                                                                                                                                                                     
  rubric/                                    # 新模块: Rubric 核心                                                                                                                          
  __init__.py                                                                                                                                                                               
  schemas.py                               # 数据结构: ExecutableRubric, EvidenceRecord                                                                                                     
  generator/                                                                                                                                                                                
  __init__.py                                                                                                                                                                               
  base.py                              # BaseRubricGenerator 接口                                                                                                                           
  llm_generator.py                     # LLM 驱动 (GPT-4o/Claude)                                                                                                                           
  executor/                                                                                                                                                                                 
  __init__.py                                                                                                                                                                               
  base.py                              # BaseRubricExecutor 接口                                                                                                                            
  default_executor.py                  # 默认执行器                                                                                                                                         
  primitives.py                        # 评分原语                                                                                                                                           
                                                                                                                                                                                            
  experimental/dataset/                                                                                                                                                                     
  latency_bucketed_sampler.py              # 延迟分桶采样器 (新)                                                                                                                            
                                                                                                                                                                                            
  interactions/                                                                                                                                                                             
  rubric_interaction.py                    # Rubric 交互 (新)                                                                                                                               
                                                                                                                                                                                            
  workers/reward_manager/                                                                                                                                                                   
  rubric_manager.py                        # Rubric 奖励管理器 (新)                                                                                                                         
                                                                                                                                                                                            
  examples/agentic_rubric/                     # 示例                                                                                                                                       
  config/                                                                                                                                                                                   
  rubric_rl_grpo.yaml                      # 训练配置                                                                                                                                       
  tool_config/                                                                                                                                                                              
  verification_tools.yaml                # 工具配置                                                                                                                                         
  data_preprocess/                                                                                                                                                                          
  rubric_dataset.py                        # 离线合成脚本                                                                                                                                   
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 核心数据结构                                                                                                                                                                           
                                                                                                                                                                                            
  ### ExecutableRubric $\rho = \langle \mathcal{V}, \Phi, \mathsf{Agg} \rangle$                                                                                                             
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  # verl/rubric/schemas.py                                                                                                                                                                  
  class VerificationItem(BaseModel):                                                                                                                                                        
  id: str                                                                                                                                                                                   
  description: str               # 验证项描述                                                                                                                                               
  category: str                  # "correctness", "style", "safety"                                                                                                                         
  weight: float = 1.0                                                                                                                                                                       
  is_gate: bool = False          # 硬门控 (失败则总分为0)                                                                                                                                   
                                                                                                                                                                                            
  class EvidencePlan(BaseModel):                                                                                                                                                            
  check_id: str                  # 关联的验证项 ID                                                                                                                                          
  tool_name: str                 # 调用的工具                                                                                                                                               
  argument_extractor: str        # 从 y/τ 提取参数的方式                                                                                                                                    
  timeout_ms: int = 30000                                                                                                                                                                   
                                                                                                                                                                                            
  class AggregationRule(BaseModel):                                                                                                                                                         
  method: Literal["weighted_sum", "min", "product"]                                                                                                                                         
  normalize: bool = True                                                                                                                                                                    
                                                                                                                                                                                            
  class ExecutableRubric(BaseModel):                                                                                                                                                        
  id: str                                                                                                                                                                                   
  task_intent: str                                  # 从 x 提取的意图                                                                                                                       
  verification_checklist: list[VerificationItem]   # V                                                                                                                                      
  evidence_plans: list[EvidencePlan]               # Φ                                                                                                                                      
  aggregation: AggregationRule                     # Agg                                                                                                                                    
  estimated_latency_ms: int = 0                    # 用于分桶                                                                                                                               
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ### 数据集存储 (via extra_info)                                                                                                                                                           
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  {                                                                                                                                                                                         
  "prompt": [...],                                                                                                                                                                          
  "extra_info": {                                                                                                                                                                           
  "rubric": {                              # ExecutableRubric 序列化                                                                                                                        
  "id": "rubric_001",                                                                                                                                                                       
  "task_intent": "...",                                                                                                                                                                     
  "verification_checklist": [...],                                                                                                                                                          
  "evidence_plans": [...],                                                                                                                                                                  
  "aggregation": {...},                                                                                                                                                                     
  "estimated_latency_ms": 5000                                                                                                                                                              
  },                                                                                                                                                                                        
  "latency_bucket": 2,                     # 预计算的桶 ID (0-4)                                                                                                                            
  "interaction_kwargs": {"name": "rubric"}                                                                                                                                                  
  }                                                                                                                                                                                         
  }                                                                                                                                                                                         
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 关键组件                                                                                                                                                                               
                                                                                                                                                                                            
  ### 1. Rubric Generator (离线，两阶段)                                                                                                                                                    
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  # verl/rubric/generator/llm_generator.py                                                                                                                                                  
  class LLMRubricGenerator(BaseRubricGenerator):                                                                                                                                            
  """两阶段 Rubric 生成器"""                                                                                                                                                                
                                                                                                                                                                                            
  async def generate(                                                                                                                                                                       
  self,                                                                                                                                                                                     
  task_input: str,            # x                                                                                                                                                           
  specification: str | None,   # 可选的规格说明                                                                                                                                             
  tool_schemas: list[dict],    # T_e^rub 的模式                                                                                                                                             
  ) -> ExecutableRubric:                                                                                                                                                                    
                                                                                                                                                                                            
  # Stage 1: 意图提取 (只看 x 和 spec)                                                                                                                                                      
  direction = await self._extract_intent(task_input, specification)                                                                                                                         
                                                                                                                                                                                            
  # Stage 2: 工具编译 (看 x, direction, tools)                                                                                                                                              
  rubric = await self._compile_to_executable(                                                                                                                                               
  task_input, direction, tool_schemas                                                                                                                                                       
  )                                                                                                                                                                                         
                                                                                                                                                                                            
  # 延迟估计                                                                                                                                                                                
  rubric.estimated_latency_ms = self._estimate_latency(rubric)                                                                                                                              
  return rubric                                                                                                                                                                             
                                                                                                                                                                                            
  async def _extract_intent(self, x: str, spec: str | None) -> RubricDirection:                                                                                                             
  """Stage 1: 从任务中提取验证意图"""                                                                                                                                                       
  # 输出: 需要验证什么 (正确性/效率/安全性等)                                                                                                                                               
  pass                                                                                                                                                                                      
                                                                                                                                                                                            
  async def _compile_to_executable(                                                                                                                                                         
  self, x: str, direction: RubricDirection, tools: list[dict]                                                                                                                               
  ) -> ExecutableRubric:                                                                                                                                                                    
  """Stage 2: 编译为可执行 rubric"""                                                                                                                                                        
  # 输出: 具体的验证项 + 证据计划                                                                                                                                                           
  pass                                                                                                                                                                                      
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ### 2. Rubric Executor (在线)                                                                                                                                                             
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  # verl/rubric/executor/default_executor.py                                                                                                                                                
  class DefaultRubricExecutor(BaseRubricExecutor):                                                                                                                                          
  """执行 rubric，输入 (x, y, τ, ρ)"""                                                                                                                                                      
                                                                                                                                                                                            
  async def execute(                                                                                                                                                                        
  self,                                                                                                                                                                                     
  task_input: str,           # x                                                                                                                                                            
  policy_output: str,        # y                                                                                                                                                            
  policy_trace: list[dict],  # τ (policy 的工具调用记录)                                                                                                                                    
  rubric: ExecutableRubric,  # ρ                                                                                                                                                            
  tools: dict[str, BaseTool] # T_e^rub                                                                                                                                                      
  ) -> tuple[float, list[EvidenceRecord], dict]:                                                                                                                                            
                                                                                                                                                                                            
  # 1. 收集证据 (并行查询 T_e^rub)                                                                                                                                                          
  evidence_records = await self._collect_evidence(                                                                                                                                          
  task_input, policy_output, policy_trace, rubric, tools                                                                                                                                    
  )                                                                                                                                                                                         
                                                                                                                                                                                            
  # 2. 评分每个验证项                                                                                                                                                                       
  # score_j = φ_j(y, e_j)                                                                                                                                                                   
  check_scores = []                                                                                                                                                                         
  for check in rubric.verification_checklist:                                                                                                                                               
  evidence = self._get_evidence_for_check(evidence_records, check.id)                                                                                                                       
  score = self._score_check(policy_output, evidence, check)                                                                                                                                 
  check_scores.append((check, score))                                                                                                                                                       
                                                                                                                                                                                            
  # 3. 聚合 (处理门控)                                                                                                                                                                      
  # R(x,y) = Agg({score_j}) with gates                                                                                                                                                      
  reward = self._aggregate_with_gates(rubric.aggregation, check_scores)                                                                                                                     
                                                                                                                                                                                            
  return reward, evidence_records, {"check_scores": check_scores}                                                                                                                           
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ### 3. Latency-Bucketed Sampler (时间分桶)                                                                                                                                                
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  # verl/experimental/dataset/latency_bucketed_sampler.py                                                                                                                                   
  class LatencyBucketedSampler(AbstractCurriculumSampler):                                                                                                                                  
  """按延迟分组以提高批处理效率"""                                                                                                                                                          
                                                                                                                                                                                            
  def __init__(self, data_source, data_config, num_buckets=5):                                                                                                                              
  # 按 extra_info["latency_bucket"] 分组索引                                                                                                                                                
  self.bucket_indices = defaultdict(list)                                                                                                                                                   
  for idx in range(len(data_source)):                                                                                                                                                       
  bucket = data_source[idx]["extra_info"]["latency_bucket"]                                                                                                                                 
  self.bucket_indices[bucket].append(idx)                                                                                                                                                   
                                                                                                                                                                                            
  def __iter__(self):                                                                                                                                                                       
  # 按桶顺序返回索引                                                                                                                                                                        
  for bucket in sorted(self.bucket_indices.keys()):                                                                                                                                         
  yield from self.bucket_indices[bucket]                                                                                                                                                    
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ### 4. Rubric Reward Manager (集成)                                                                                                                                                       
                                                                                                                                                                                            
  ```python                                                                                                                                                                                 
  # verl/workers/reward_manager/rubric_manager.py                                                                                                                                           
  @register("rubric")                                                                                                                                                                       
  class RubricRewardManager(AbstractRewardManager):                                                                                                                                         
  """与 verl 训练循环集成"""                                                                                                                                                                
                                                                                                                                                                                            
  def __call__(self, data: DataProto, return_dict=False):                                                                                                                                   
  reward_tensor = torch.zeros_like(data.batch["responses"])                                                                                                                                 
                                                                                                                                                                                            
  for i in range(len(data)):                                                                                                                                                                
  rubric = ExecutableRubric(**data.non_tensor_batch["extra_info"][i]["rubric"])                                                                                                             
  response = self._decode_response(data, i)                                                                                                                                                 
                                                                                                                                                                                            
  # 执行 rubric                                                                                                                                                                             
  reward, _, _ = await self.executor.execute(rubric, ...)                                                                                                                                   
  reward_tensor[i, -1] = reward                                                                                                                                                             
                                                                                                                                                                                            
  return reward_tensor                                                                                                                                                                      
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 实现步骤 (按 Algorithm 1)                                                                                                                                                              
                                                                                                                                                                                            
  ### Phase 1: 离线数据准备                                                                                                                                                                 
                                                                                                                                                                                            
  **脚本**: `examples/agentic_rubric/data_preprocess/rubric_dataset.py`                                                                                                                     
                                                                                                                                                                                            
  ```bash                                                                                                                                                                                   
  python rubric_dataset.py \                                                                                                                                                                
  --input_path data/raw_tasks.parquet \                                                                                                                                                     
  --output_path data/tasks_with_rubrics.parquet \                                                                                                                                           
  --tool_config config/tool_config/verification_tools.yaml \                                                                                                                                
  --generator_model gpt-4o \                                                                                                                                                                
  --num_buckets 5                                                                                                                                                                           
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  **流程**:                                                                                                                                                                                 
  1. 加载工具模式 $\mathcal{T}_e^{rub}$                                                                                                                                                     
  2. 为每个 $x_i$ 生成 $\rho_i \leftarrow \mathcal{G}_\psi(x_i, \mathcal{T}_e^{rub})$                                                                                                       
  3. 估计延迟 $L_i$ 并分配桶                                                                                                                                                                
  4. 保存增强数据集 $\mathcal{D}' = \{(x_i, \rho_i, L_i)\}$                                                                                                                                 
                                                                                                                                                                                            
  ### Phase 2: 在线训练                                                                                                                                                                     
                                                                                                                                                                                            
  **配置**: `examples/agentic_rubric/config/rubric_rl_grpo.yaml`                                                                                                                            
                                                                                                                                                                                            
  ```yaml                                                                                                                                                                                   
  data:                                                                                                                                                                                     
  train_files: data/tasks_with_rubrics.parquet                                                                                                                                              
  sampler:                                                                                                                                                                                  
  class_path: verl.experimental.dataset.latency_bucketed_sampler                                                                                                                            
  class_name: LatencyBucketedSampler                                                                                                                                                        
  kwargs:                                                                                                                                                                                   
  num_buckets: 5                                                                                                                                                                            
                                                                                                                                                                                            
  reward_manager:                                                                                                                                                                           
  source: register                                                                                                                                                                          
  name: rubric                                                                                                                                                                              
                                                                                                                                                                                            
  actor_rollout_ref:                                                                                                                                                                        
  rollout:                                                                                                                                                                                  
  multi_turn:                                                                                                                                                                               
  enable: True                                                                                                                                                                              
  tool_config_path: config/tool_config/policy_tools.yaml                                                                                                                                    
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  **训练循环** (伪代码):                                                                                                                                                                    
  ```                                                                                                                                                                                       
  for batch in dataloader (latency-bucketed):                                                                                                                                               
  # 1. Policy rollout                                                                                                                                                                       
  (y, τ) ~ π_θ(x; T_e^pol)                                                                                                                                                                  
                                                                                                                                                                                            
  # 2. Rubric execution (parallel)                                                                                                                                                          
  for (x, y, τ, ρ) in parallel:                                                                                                                                                             
  E, r = R(x, y, τ, ρ; T_e^rub)                                                                                                                                                             
                                                                                                                                                                                            
  # 3. PPO/GRPO update                                                                                                                                                                      
  θ ← θ + ∇J(θ)                                                                                                                                                                             
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 新文件清单                                                                                                                                                                             
                                                                                                                                                                                            
  | 文件 | 说明 | 优先级 |                                                                                                                                                                  
  |------|------|--------|                                                                                                                                                                  
  | `verl/rubric/__init__.py` | 模块入口 | P0 |                                                                                                                                             
  | `verl/rubric/schemas.py` | 数据结构定义 | P0 |                                                                                                                                          
  | `verl/rubric/generator/base.py` | Generator 接口 | P0 |                                                                                                                                 
  | `verl/rubric/generator/llm_generator.py` | LLM 实现 | P0 |                                                                                                                              
  | `verl/rubric/executor/base.py` | Executor 接口 | P0 |                                                                                                                                   
  | `verl/rubric/executor/default_executor.py` | 默认执行器 | P0 |                                                                                                                          
  | `verl/rubric/executor/primitives.py` | 评分原语 | P0 |                                                                                                                                  
  | `verl/workers/reward_manager/rubric_manager.py` | 奖励管理器 | P0 |                                                                                                                     
  | `verl/experimental/dataset/latency_bucketed_sampler.py` | 分桶采样器 | P0 |                                                                                                             
  | `examples/agentic_rubric/data_preprocess/rubric_dataset.py` | 数据预处理 | P1 |                                                                                                         
  | `examples/agentic_rubric/config/*.yaml` | 配置文件 | P1 |                                                                                                                               
                                                                                                                                                                                            
  ## 验证方式                                                                                                                                                                               
                                                                                                                                                                                            
  ### 单元测试                                                                                                                                                                              
  ```bash                                                                                                                                                                                   
  pytest tests/rubric/test_schemas.py                                                                                                                                                       
  pytest tests/rubric/test_executor.py                                                                                                                                                      
  pytest tests/rubric/test_sampler.py                                                                                                                                                       
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ### 端到端测试                                                                                                                                                                            
  ```bash                                                                                                                                                                                   
  # 数据预处理                                                                                                                                                                              
  python examples/agentic_rubric/data_preprocess/rubric_dataset.py ...                                                                                                                      
                                                                                                                                                                                            
  # 训练                                                                                                                                                                                    
  python -m verl.trainer.main_ppo --config-name rubric_rl_grpo                                                                                                                              
  ```                                                                                                                                                                                       
                                                                                                                                                                                            
  ---                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 待澄清问题                                                                                                                                                                             
                                                                                                                                                                                            
  以下问题需要您的确认以完善计划:                                                                                                                                                           
                                                                                                                                                                                            
  1. **Rubric Generator 模型**: 使用哪个 LLM 生成 rubric？(GPT-4o / Claude / 本地模型)                                                                                                      
                                                                                                                                                                                            
  2. **目标任务/数据集**: 您计划在哪个任务上验证框架？(代码生成 / 数学推理 / 其他)                                                                                                          
                                                                                                                                                                                            
  3. **验证工具优先级**: 需要哪些验证工具？                                                                                                                                                 
  - 代码执行 (SandboxFusion) - verl 已支持                                                                                                                                                  
  - 搜索/检索 (SearchTool) - verl 已支持                                                                                                                                                    
  - 数据库查询 - 需要新实现                                                                                                                                                                 
  - 其他自定义工具                                                                                                                                                                          
                                                                                                                                                                                            
  4. **在线生成需求**: 是否需要支持训练时动态生成 rubric？还是全部离线预生成？                                                                                                              
                                                                                                                                                                                            
  ---                                                                                                                                                                                       
                                                                                                                                                                                            
  ## 用户确认的设置                                                                                                                                                                         
                                                                                                                                                                                            
  | 配置项 | 选择 |                                                                                                                                                                         
  |--------|------|                                                                                                                                                                         
  | **Rubric Generator** | GPT-4o |                                                                                                                                                         
  | **目标任务** | 代码生成 (使用 SandboxFusion 验证) |                                                                                                                                     
  | **生成模式** | 仅离线预生成 |                                                                                                                                                           
                                                                                                                                                                                            
  基于以上选择，实现将聚焦于：                                                                                                                                                              
  - 代码执行验证 (SandboxFusion 工具)                                                                                                                                                       
  - 测试用例验证                                                                                                                                                                            
  - 离线 rubric 合成管道                         