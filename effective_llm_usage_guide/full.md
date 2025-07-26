# Complete LLM Mastery Guide: From Basics to Advanced AI Systems

## Table of Contents
1. [Learning Path Overview](#learning-path-overview)
2. [Level 1: Prompt Engineering Fundamentals](#level-1-prompt-engineering-fundamentals)
3. [Level 2: Advanced Prompting Techniques](#level-2-advanced-prompting-techniques)
4. [Level 3: Context Engineering & Memory](#level-3-context-engineering--memory)
5. [Level 4: Tool Integration & Basic RAG](#level-4-tool-integration--basic-rag)
6. [Level 5: Advanced RAG & Knowledge Systems](#level-5-advanced-rag--knowledge-systems)
7. [Level 6: Single Agent Workflows](#level-6-single-agent-workflows)
8. [Level 7: Multi-Agent Coordination](#level-7-multi-agent-coordination)
9. [Level 8: Production-Grade Systems](#level-8-production-grade-systems)
10. [Working with Small LLM Models](#working-with-small-llm-models)
11. [Capability Progression Summary](#capability-progression-summary)

---

## Learning Path Overview

This guide follows a structured progression where each level builds upon the previous, unlocking new capabilities:

```
Level 1: Basic Prompts → Reliable single responses
Level 2: Advanced Prompts → Complex reasoning & format control
Level 3: Context Engineering → Handling large information sets
Level 4: Tool Integration → Interaction with external systems
Level 5: Advanced RAG → Dynamic knowledge retrieval
Level 6: Agent Workflows → Autonomous task execution
Level 7: Multi-Agent → Collaborative AI systems
Level 8: Production → Scalable, robust applications
```

**Prerequisites:** Basic programming knowledge, familiarity with APIs

---

## Level 1: Prompt Engineering Fundamentals

### What You'll Learn
- How to write clear, effective prompts
- Basic structure and formatting
- Common pitfalls and how to avoid them

### What You Can Build After This Level
✅ Reliable content generation tools  
✅ Simple classification systems  
✅ Basic question-answering applications  
✅ Text summarization utilities  

### 1.1 The Anatomy of a Good Prompt

A well-structured prompt has these components:

```
[Role/Context] + [Task] + [Format] + [Constraints]
```

**Example Evolution:**

❌ **Poor:** "Write about dogs"

⚠️ **Better:** "Write an article about dogs"

✅ **Good:** 
```
You are a pet care expert writing for new dog owners.

Task: Write a 300-word article about golden retriever care
Format: Include sections for feeding, exercise, and grooming
Constraints: Use simple language, include 2-3 specific tips per section
```

### 1.2 Core Principles

#### Principle 1: Be Specific
```python
# Generic prompt
prompt = "Help me with my presentation"

# Specific prompt
prompt = """
I'm giving a 10-minute presentation to executives about our Q3 sales performance.
Create an outline with:
- 3 main points
- Supporting data for each point
- A clear call-to-action
Focus on ROI and growth metrics.
"""
```

#### Principle 2: Provide Context
```python
# Without context
prompt = "Is this a good strategy?"

# With context
prompt = """
Context: We're a SaaS startup with 50 employees, $2M ARR, competing with established players.
Strategy: Focus entirely on enterprise clients and abandon SMB market.
Analyze: Is this a good strategy? Consider our resources, market position, and growth goals.
"""
```

#### Principle 3: Control Output Format
```python
# Uncontrolled output
prompt = "Compare these products"

# Controlled format
prompt = """
Compare Product A vs Product B using this format:

**Product A:**
- Pros: [list 3]
- Cons: [list 3]
- Best for: [target user]

**Product B:**
- Pros: [list 3]  
- Cons: [list 3]
- Best for: [target user]

**Recommendation:** [Which to choose and why]
"""
```

### 1.3 Practical Exercise: Building Your First Application

Let's build a simple email classifier:

```python
def classify_email(email_content):
    prompt = f"""
    You are an email classification system.
    
    Task: Classify this email into one category
    Categories: urgent, important, spam, newsletter, personal
    
    Email content: {email_content}
    
    Respond with only the category name in lowercase.
    """
    
    response = llm.generate(prompt)
    return response.strip().lower()

# Test it
email = "URGENT: Your account will be suspended unless you click this link immediately!"
category = classify_email(email)
print(f"Category: {category}")  # Should output: spam
```

### 1.4 Common Beginner Mistakes

**Mistake 1: Vague Instructions**
```python
❌ "Make this better"
✅ "Improve readability by: shortening sentences, adding bullet points, removing jargon"
```

**Mistake 2: No Examples**
```python
❌ "Format as JSON"
✅ "Format as JSON like this: {'name': 'John', 'age': 30, 'city': 'NYC'}"
```

**Mistake 3: Assuming Context**
```python
❌ "What should we do next?"
✅ "Given our product launch failed, what should our startup do next? Consider pivoting, fundraising, or cost-cutting."
```

---

## Level 2: Advanced Prompting Techniques

### What You'll Learn
- Chain-of-thought reasoning
- Few-shot learning patterns
- Role-based prompting
- Conditional logic in prompts

### What You Can Build After This Level
✅ Complex reasoning applications  
✅ Multi-step problem solvers  
✅ Adaptive response systems  
✅ Domain expert simulators  

### 2.1 Chain-of-Thought (CoT) Prompting

CoT makes AI show its reasoning process, dramatically improving accuracy on complex tasks.

**Basic CoT:**
```python
def solve_math_problem(problem):
    prompt = f"""
    Solve this step-by-step:
    
    Problem: {problem}
    
    Let me think through this step by step:
    1. [First, I need to...]
    2. [Then, I should...]
    3. [Finally, I can...]
    
    Step-by-step solution:
    """
    return llm.generate(prompt)

# Example usage
problem = "A store sells apples for $2 per pound. If I buy 3.5 pounds and pay with a $10 bill, how much change will I receive?"
solution = solve_math_problem(problem)
```

**Advanced CoT with Self-Correction:**
```python
def solve_with_verification(problem):
    prompt = f"""
    Problem: {problem}
    
    Step 1 - Initial Solution:
    [Solve the problem step by step]
    
    Step 2 - Verification:
    [Check my work by using a different approach]
    
    Step 3 - Final Answer:
    [Confirm or correct my initial solution]
    """
    return llm.generate(prompt)
```

### 2.2 Few-Shot Learning

Teach the AI by example rather than explanation.

**Progressive Examples:**
```python
def create_few_shot_classifier():
    prompt = """
    Classify the sentiment of customer feedback:
    
    Example 1:
    Feedback: "The product arrived quickly and works perfectly!"
    Sentiment: Positive
    Reasoning: Expresses satisfaction with delivery and quality
    
    Example 2:
    Feedback: "Delivery was delayed and the item was damaged"
    Sentiment: Negative  
    Reasoning: Reports problems with service and product condition
    
    Example 3:
    Feedback: "The order confirmation was received"
    Sentiment: Neutral
    Reasoning: Factual statement without emotional content
    
    Now classify this feedback:
    Feedback: "{feedback}"
    Sentiment:
    Reasoning:
    """
    return prompt

def classify_sentiment(feedback):
    prompt = create_few_shot_classifier().format(feedback=feedback)
    return llm.generate(prompt)
```

### 2.3 Role-Based Prompting

Make the AI adopt specific expertise and perspectives.

```python
class ExpertRoles:
    @staticmethod
    def financial_advisor():
        return """
        You are a certified financial advisor with 15 years of experience.
        Your expertise: retirement planning, tax optimization, risk management
        Your approach: Conservative, evidence-based, always consider the client's risk tolerance
        Your communication style: Clear explanations, specific actionable steps, warn about risks
        """
    
    @staticmethod
    def technical_architect():
        return """
        You are a senior technical architect at a Fortune 500 company.
        Your expertise: Scalable systems, cloud architecture, performance optimization
        Your approach: Consider trade-offs, think about maintenance, prioritize security
        Your communication style: Technical but accessible, provide alternatives, mention potential issues
        """
    
    @staticmethod
    def marketing_strategist():
        return """
        You are a marketing strategist who has launched 50+ successful campaigns.
        Your expertise: Brand positioning, customer psychology, multi-channel campaigns
        Your approach: Data-driven, customer-centric, test-and-iterate mindset
        Your communication style: Creative but grounded, provide metrics for success, consider budget constraints
        """

def get_expert_advice(role, question):
    role_prompt = getattr(ExpertRoles, role)()
    
    prompt = f"""
    {role_prompt}
    
    Question: {question}
    
    Provide your expert advice:
    """
    
    return llm.generate(prompt)

# Usage
advice = get_expert_advice("financial_advisor", "I'm 25 with $10k savings. How should I start investing?")
```

### 2.4 Tree-of-Thought (ToT) Prompting

Advanced technique that enables models to explore multiple reasoning paths simultaneously.

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

class TreeOfThoughtPrompting:
    def __init__(self, llm):
        self.llm = llm
        
    def generate_reasoning_paths(self, problem: str, num_paths: int = 3):
        """Generate multiple reasoning approaches"""
        
        tot_prompt = PromptTemplate(
            input_variables=["problem", "step"],
            template="""
            Problem: {problem}
            
            Current step: {step}
            
            Generate 3 different approaches to solve this step:
            Approach 1: [Describe first approach]
            Approach 2: [Describe second approach] 
            Approach 3: [Describe third approach]
            
            Evaluate each approach:
            - Feasibility: [Rate 1-10]
            - Accuracy potential: [Rate 1-10]
            - Reasoning: [Explain evaluation]
            
            Select the best approach and continue.
            """
        )
        
        return tot_prompt
    
    def solve_with_tree_search(self, problem: str, max_depth: int = 3):
        """Solve problem using tree-of-thought search"""
        
        reasoning_tree = {
            "problem": problem,
            "paths": [],
            "final_solution": None
        }
        
        # Generate multiple reasoning paths
        for depth in range(max_depth):
            path_prompt = f"""
            Problem: {problem}
            Depth: {depth + 1}
            
            Generate reasoning step {depth + 1}:
            Consider multiple approaches and select the most promising one.
            """
            
            path_result = self.llm.generate(path_prompt)
            reasoning_tree["paths"].append({
                "depth": depth + 1,
                "reasoning": path_result
            })
        
        # Synthesize final solution
        synthesis_prompt = f"""
        Problem: {problem}
        Reasoning paths explored: {reasoning_tree['paths']}
        
        Synthesize the best solution from all explored paths:
        """
        
        final_solution = self.llm.generate(synthesis_prompt)
        reasoning_tree["final_solution"] = final_solution
        
        return reasoning_tree

# Usage example
tot_solver = TreeOfThoughtPrompting(llm)
result = tot_solver.solve_with_tree_search(
    "Design a sustainable urban transportation system for a city of 2 million people"
)
```

### 2.5 Self-Consistency Prompting

Generate multiple reasoning paths and select the most consistent answer.

```python
from collections import Counter
import asyncio

class SelfConsistencyPrompting:
    def __init__(self, llm):
        self.llm = llm
        
    async def generate_multiple_responses(self, prompt: str, num_samples: int = 5):
        """Generate multiple responses for consistency checking"""
        
        tasks = []
        for i in range(num_samples):
            # Add slight variation to reduce deterministic responses
            varied_prompt = f"{prompt}\n\n[Attempt {i+1}: Think through this carefully]"
            tasks.append(self.llm.generate(varied_prompt))
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    def find_consistent_answer(self, responses: list) -> dict:
        """Find most consistent answer from multiple responses"""
        
        # Extract key conclusions from each response
        conclusions = []
        for response in responses:
            conclusion = self.extract_conclusion(response)
            conclusions.append(conclusion)
        
        # Find most common conclusion
        conclusion_counts = Counter(conclusions)
        most_common = conclusion_counts.most_common(1)[0]
        
        return {
            "final_answer": most_common[0],
            "confidence": most_common[1] / len(responses),
            "all_responses": responses,
            "consistency_analysis": conclusion_counts
        }
    
    def extract_conclusion(self, response: str) -> str:
        """Extract main conclusion from response"""
        # Simple extraction - look for conclusion indicators
        conclusion_markers = ["conclusion:", "therefore:", "in summary:", "final answer:"]
        
        for marker in conclusion_markers:
            if marker in response.lower():
                parts = response.lower().split(marker)
                if len(parts) > 1:
                    return parts[1].strip()[:100]  # First 100 chars of conclusion
        
        # If no markers, return last sentence
        sentences = response.split('.')
        return sentences[-1].strip() if sentences else response[:100]

# Advanced usage with LangChain
async def self_consistent_problem_solving(problem: str):
    consistency_prompter = SelfConsistencyPrompting(llm)
    
    # Generate multiple solutions
    responses = await consistency_prompter.generate_multiple_responses(
        f"Solve this problem step by step: {problem}",
        num_samples=5
    )
    
    # Find most consistent solution
    result = consistency_prompter.find_consistent_answer(responses)
    
    return result
```

### 2.6 Dynamic Example Selection

Intelligently select few-shot examples based on query similarity.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

class DynamicFewShotPrompting:
    def __init__(self, example_bank: list):
        self.example_bank = example_bank
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._build_example_index()
        
    def _build_example_index(self):
        """Build vector index of examples"""
        example_texts = [ex["input"] for ex in self.example_bank]
        self.vector_store = FAISS.from_texts(example_texts, self.embeddings)
    
    def select_relevant_examples(self, query: str, k: int = 3) -> list:
        """Select most relevant examples for the query"""
        
        # Find similar examples
        similar_docs = self.vector_store.similarity_search(query, k=k)
        
        # Get full examples
        relevant_examples = []
        for doc in similar_docs:
            # Find corresponding example
            for example in self.example_bank:
                if example["input"] == doc.page_content:
                    relevant_examples.append(example)
                    break
        
        return relevant_examples
    
    def create_few_shot_prompt(self, query: str, task_description: str) -> str:
        """Create optimized few-shot prompt"""
        
        relevant_examples = self.select_relevant_examples(query, k=3)
        
        prompt_parts = [task_description, "\nHere are some examples:\n"]
        
        for i, example in enumerate(relevant_examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            if "reasoning" in example:
                prompt_parts.append(f"Reasoning: {example['reasoning']}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"Now solve this:",
            f"Input: {query}",
            f"Output:"
        ])
        
        return "\n".join(prompt_parts)

# Example usage
example_bank = [
    {
        "input": "Classify: 'This product is absolutely terrible!'",
        "output": "Negative",
        "reasoning": "Contains strong negative language ('terrible') with emphasis ('absolutely')"
    },
    {
        "input": "Classify: 'Great quality and fast delivery!'", 
        "output": "Positive",
        "reasoning": "Expresses satisfaction with product quality and service"
    },
    {
        "input": "Classify: 'The package arrived today.'",
        "output": "Neutral", 
        "reasoning": "Factual statement without emotional content"
    }
]

few_shot_prompter = DynamicFewShotPrompting(example_bank)
prompt = few_shot_prompter.create_few_shot_prompt(
    "The item was okay, nothing special but functional",
    "Classify the sentiment of customer feedback as Positive, Negative, or Neutral."
)
```

### 2.5 Template System for Reusable Prompts

```python
class PromptTemplates:
    ANALYSIS_TEMPLATE = """
    Role: You are a {expert_type} with {years_experience} years of experience.
    
    Task: Analyze the following {subject_type}:
    {content}
    
    Analysis Framework:
    1. Strengths: {strength_focus}
    2. Weaknesses: {weakness_focus}  
    3. Opportunities: {opportunity_focus}
    4. Threats/Risks: {risk_focus}
    5. Recommendations: {recommendation_focus}
    
    Format: Provide 2-3 bullet points for each section.
    Tone: {tone}
    """
    
    CREATIVE_TEMPLATE = """
    You are a creative {role} known for {style_traits}.
    
    Project: Create {deliverable_type} for {target_audience}
    Requirements:
    - Theme: {theme}
    - Mood: {mood}
    - Constraints: {constraints}
    - Length: {length}
    
    Creative brief: {brief}
    
    Deliverable:
    """
    
    TECHNICAL_TEMPLATE = """
    Role: Senior {technical_role} specializing in {specialization}
    
    Challenge: {technical_challenge}
    
    Context:
    - Current setup: {current_state}
    - Requirements: {requirements}
    - Constraints: {constraints}
    - Performance targets: {performance_targets}
    
    Provide:
    1. Technical approach
    2. Implementation steps  
    3. Potential issues and mitigation
    4. Alternative solutions
    """

def use_template(template_name, **kwargs):
    template = getattr(PromptTemplates, template_name)
    return template.format(**kwargs)

# Usage
analysis_prompt = use_template(
    "ANALYSIS_TEMPLATE",
    expert_type="marketing strategist",
    years_experience="10",
    subject_type="product launch campaign",
    content="Our new AI-powered productivity app launching next month...",
    strength_focus="market positioning and unique value prop",
    weakness_focus="resource constraints and competition",
    opportunity_focus="market trends and partnerships",
    risk_focus="timing and user adoption",
    recommendation_focus="immediate action items",
    tone="professional but actionable"
)
```

---

## Level 3: Context Engineering & Memory

### What You'll Learn
- Managing large amounts of information
- Context window optimization
- Memory systems for conversational AI
- Information prioritization techniques

### What You Can Build After This Level
✅ Long-conversation chatbots  
✅ Document analysis systems  
✅ Multi-turn reasoning applications  
✅ Personalized AI assistants  

### 3.1 Understanding Context Limitations

Every LLM has a context window (token limit). Here's how to work within it:

```python
import tiktoken

class ContextManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.reserve_tokens = 500  # Reserve for response
        
    def count_tokens(self, text):
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_to_limit(self, text, max_tokens=None):
        """Truncate text to fit token limit"""
        max_tokens = max_tokens or (self.max_tokens - self.reserve_tokens)
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def smart_truncate(self, text, preserve_end=True):
        """Intelligently truncate preserving important parts"""
        if self.count_tokens(text) <= self.max_tokens - self.reserve_tokens:
            return text
            
        if preserve_end:
            # Keep the end of the text (often most relevant)
            tokens = self.encoding.encode(text)
            keep_tokens = self.max_tokens - self.reserve_tokens
            truncated = tokens[-keep_tokens:]
            return self.encoding.decode(truncated)
        else:
            return self.truncate_to_limit(text)

# Usage
context_mgr = ContextManager()
long_document = "..." # Your long text
optimized_text = context_mgr.smart_truncate(long_document)
```

### 3.2 Information Hierarchy and Prioritization

```python
class InformationPrioritizer:
    def __init__(self):
        self.priority_weights = {
            "critical": 10,
            "high": 7,
            "medium": 5,
            "low": 2,
            "background": 1
        }
    
    def prioritize_content(self, content_blocks):
        """
        content_blocks: List of dicts with 'content', 'priority', and 'relevance_score'
        """
        scored_blocks = []
        
        for block in content_blocks:
            priority_score = self.priority_weights.get(block['priority'], 1)
            relevance_score = block.get('relevance_score', 0.5)
            total_score = priority_score * relevance_score
            
            scored_blocks.append({
                **block,
                'total_score': total_score
            })
        
        # Sort by total score (highest first)
        return sorted(scored_blocks, key=lambda x: x['total_score'], reverse=True)
    
    def build_context(self, prioritized_blocks, max_tokens=3000):
        """Build context string within token limits"""
        context_mgr = ContextManager(max_tokens=max_tokens)
        context_parts = []
        current_tokens = 0
        
        for block in prioritized_blocks:
            block_tokens = context_mgr.count_tokens(block['content'])
            
            if current_tokens + block_tokens <= max_tokens:
                context_parts.append(f"[{block['priority'].upper()}] {block['content']}")
                current_tokens += block_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if we have meaningful space
                    truncated = context_mgr.truncate_to_limit(block['content'], remaining_tokens)
                    context_parts.append(f"[{block['priority'].upper()}] {truncated}...")
                break
        
        return "\n\n".join(context_parts)

# Example usage
content_blocks = [
    {
        "content": "User's recent purchases show preference for premium products",
        "priority": "high",
        "relevance_score": 0.9
    },
    {
        "content": "Company founded in 1995, headquarters in Seattle",
        "priority": "background", 
        "relevance_score": 0.3
    },
    {
        "content": "Customer support ticket: payment issue needs immediate resolution",
        "priority": "critical",
        "relevance_score": 1.0
    }
]

prioritizer = InformationPrioritizer()
prioritized = prioritizer.prioritize_content(content_blocks)
context = prioritizer.build_context(prioritized)
```

### 3.3 Conversational Memory System

```python
class ConversationMemory:
    def __init__(self, max_history=10, context_manager=None):
        self.messages = []
        self.max_history = max_history
        self.context_mgr = context_manager or ContextManager()
        self.summary = ""
        
    def add_message(self, role, content, metadata=None):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Manage memory size
        if len(self.messages) > self.max_history * 2:  # *2 for user+assistant pairs
            self._compress_old_messages()
    
    def _compress_old_messages(self):
        """Compress old messages into summary"""
        if len(self.messages) <= self.max_history:
            return
            
        # Take oldest half of messages for compression
        to_compress = self.messages[:len(self.messages)//2]
        self.messages = self.messages[len(self.messages)//2:]
        
        # Create summary of compressed messages
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in to_compress
        ])
        
        summary_prompt = f"""
        Summarize this conversation, preserving key information and context:
        
        {conversation_text}
        
        Summary should include:
        - Main topics discussed
        - Important facts or preferences mentioned
        - Any decisions or commitments made
        - Context needed for future responses
        
        Summary:
        """
        
        new_summary = llm.generate(summary_prompt)
        
        if self.summary:
            self.summary = f"{self.summary}\n\nAdditional context: {new_summary}"
        else:
            self.summary = new_summary
    
    def get_context_for_prompt(self):
        """Get conversation context optimized for current prompt"""
        context_parts = []
        
        # Add summary if exists
        if self.summary:
            context_parts.append(f"Previous conversation context:\n{self.summary}")
        
        # Add recent messages
        recent_messages = self.messages[-self.max_history:]
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                context_parts.append(f"{msg['role']}: {msg['content']}")
        
        full_context = "\n".join(context_parts)
        
        # Ensure context fits in token limit
        return self.context_mgr.smart_truncate(full_context)
    
    def search_memory(self, query, top_k=3):
        """Search conversation history for relevant information"""
        # Simple keyword-based search (in production, use embeddings)
        query_words = query.lower().split()
        scored_messages = []
        
        for msg in self.messages:
            content_words = msg['content'].lower().split()
            score = sum(1 for word in query_words if word in content_words)
            if score > 0:
                scored_messages.append((score, msg))
        
        # Return top matches
        scored_messages.sort(reverse=True)
        return [msg for _, msg in scored_messages[:top_k]]

# Usage in a chatbot
class ContextAwareChatbot:
    def __init__(self):
        self.memory = ConversationMemory()
        
    def chat(self, user_input):
        # Add user message to memory
        self.memory.add_message("user", user_input)
        
        # Get conversation context
        context = self.memory.get_context_for_prompt()
        
        # Create prompt with context
        prompt = f"""
        {context}
        
        user: {user_input}
        
        assistant: """
        
        # Generate response
        response = llm.generate(prompt)
        
        # Add response to memory
        self.memory.add_message("assistant", response)
        
        return response

# Example usage
chatbot = ContextAwareChatbot()
response1 = chatbot.chat("My name is Alice and I love hiking")
response2 = chatbot.chat("What outdoor activities would you recommend?")  # Will remember Alice likes hiking
```

### 3.4 Advanced Memory Systems

```python
from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AdvancedMemorySystem:
    def __init__(self, llm, vector_store):
        self.llm = llm
        
        # Multiple memory types
        self.short_term_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            return_messages=True
        )
        
        self.long_term_memory = VectorStoreRetrieverMemory(
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        self.working_memory = {}
        self.episodic_memory = []  # Store important conversation episodes
        
    def update_memory(self, interaction: dict):
        """Update all memory systems"""
        
        # Store in short-term memory
        self.short_term_memory.save_context(
            {"input": interaction["query"]},
            {"output": interaction["response"]}
        )
        
        # Archive important information to long-term memory
        if self.is_important(interaction):
            self.long_term_memory.save_context(
                {"input": interaction["query"]},
                {"output": interaction["response"]}
            )
            
            # Store as episodic memory
            episode = {
                "timestamp": time.time(),
                "query": interaction["query"],
                "response": interaction["response"],
                "importance_score": self.calculate_importance(interaction),
                "context": interaction.get("context", "")
            }
            self.episodic_memory.append(episode)
    
    def get_relevant_memories(self, current_query: str, memory_types: list = None) -> str:
        """Retrieve relevant memories for current query"""
        
        if memory_types is None:
            memory_types = ["short_term", "long_term", "episodic"]
        
        memory_parts = []
        
        if "short_term" in memory_types:
            short_term = self.short_term_memory.load_memory_variables({})
            memory_parts.append(f"Recent conversation:\n{short_term}")
        
        if "long_term" in memory_types:
            long_term = self.long_term_memory.load_memory_variables({"prompt": current_query})
            memory_parts.append(f"Relevant knowledge:\n{long_term}")
        
        if "episodic" in memory_types:
            relevant_episodes = self.search_episodic_memory(current_query)
            if relevant_episodes:
                episode_text = "\n".join([
                    f"Past interaction: {ep['query']} -> {ep['response'][:100]}..."
                    for ep in relevant_episodes
                ])
                memory_parts.append(f"Similar past interactions:\n{episode_text}")
        
        return "\n\n".join(memory_parts)
    
    def is_important(self, interaction: dict) -> bool:
        """Determine if interaction should be stored in long-term memory"""
        
        importance_prompt = f"""
        Evaluate the importance of this interaction for future reference:
        
        Query: {interaction['query']}
        Response: {interaction['response']}
        
        Consider:
        1. Contains factual information worth remembering
        2. Represents a significant decision or preference
        3. Includes instructions or procedures
        4. Shows user's goals or constraints
        
        Rate importance (1-10) and explain:
        """
        
        evaluation = self.llm.generate(importance_prompt)
        
        # Simple heuristic: look for high scores
        return any(score in evaluation for score in ["8", "9", "10"])
    
    def search_episodic_memory(self, query: str, top_k: int = 3) -> list:
        """Search episodic memory for relevant past interactions"""
        
        # Simple keyword-based search (could use embeddings)
        query_words = set(query.lower().split())
        
        scored_episodes = []
        for episode in self.episodic_memory:
            episode_words = set((episode['query'] + ' ' + episode['response']).lower().split())
            overlap = len(query_words.intersection(episode_words))
            
            if overlap > 0:
                score = overlap + episode['importance_score']
                scored_episodes.append((score, episode))
        
        # Return top matches
        scored_episodes.sort(reverse=True)
        return [episode for _, episode in scored_episodes[:top_k]]

class HierarchicalContextManager:
    """Advanced context organization in layers"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.context_hierarchy = {
            "system": {"priority": 1, "content": "", "required": True},
            "user_profile": {"priority": 2, "content": "", "required": False},
            "task_context": {"priority": 3, "content": "", "required": True},
            "conversation_history": {"priority": 4, "content": "", "required": False},
            "retrieved_knowledge": {"priority": 5, "content": "", "required": False},
            "current_query": {"priority": 6, "content": "", "required": True}
        }
        
    def add_context(self, level: str, content: str):
        """Add content to specific context level"""
        if level in self.context_hierarchy:
            self.context_hierarchy[level]["content"] = content
    
    def build_optimized_context(self) -> str:
        """Build context within token limits, prioritizing by importance"""
        
        context_parts = []
        remaining_tokens = self.max_tokens
        
        # Sort by priority (lower number = higher priority)
        sorted_levels = sorted(
            self.context_hierarchy.items(), 
            key=lambda x: x[1]["priority"]
        )
        
        for level_name, level_data in sorted_levels:
            content = level_data["content"]
            if not content:
                continue
                
            content_tokens = self.count_tokens(content)
            
            # Always include required content
            if level_data["required"]:
                if content_tokens <= remaining_tokens:
                    context_parts.append(f"[{level_name.upper()}]\n{content}")
                    remaining_tokens -= content_tokens
                else:
                    # Compress required content to fit
                    compressed = self.compress_content(content, remaining_tokens)
                    context_parts.append(f"[{level_name.upper()}]\n{compressed}")
                    remaining_tokens = 0
                    break
            
            # Include optional content if space allows
            elif content_tokens <= remaining_tokens:
                context_parts.append(f"[{level_name.upper()}]\n{content}")
                remaining_tokens -= content_tokens
        
        return "\n\n".join(context_parts)
    
    def compress_content(self, content: str, target_tokens: int) -> str:
        """Compress content to fit within token limit"""
        
        # Simple compression: take first and last parts
        sentences = content.split('.')
        if len(sentences) <= 3:
            return content[:target_tokens * 4]  # Rough char estimate
        
        # Keep first and last sentences, summarize middle
        first_part = sentences[0]
        last_part = sentences[-1]
        middle_summary = "[...content compressed...]"
        
        compressed = f"{first_part}. {middle_summary} {last_part}"
        return compressed
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count"""
        return len(text.split()) * 1.3  # Rough approximation

# Usage example
def create_context_aware_agent():
    memory_system = AdvancedMemorySystem(llm, vector_store)
    context_manager = HierarchicalContextManager()
    
    def process_query(query: str, user_profile: dict = None):
        # Build hierarchical context
        context_manager.add_context("system", "You are a helpful AI assistant.")
        context_manager.add_context("current_query", query)
        
        if user_profile:
            context_manager.add_context("user_profile", str(user_profile))
        
        # Get relevant memories
        relevant_memories = memory_system.get_relevant_memories(query)
        context_manager.add_context("conversation_history", relevant_memories)
        
        # Build final context
        full_context = context_manager.build_optimized_context()
        
        # Generate response
        response = llm.generate(f"{full_context}\n\nResponse:")
        
        # Update memory
        memory_system.update_memory({
            "query": query,
            "response": response,
            "context": full_context
        })
        
        return response
    
    return process_query
```

---

## Level 4: Tool Integration & Basic RAG

### What You'll Learn
- Connecting LLMs to external systems
- Function calling and tool use
- Basic Retrieval Augmented Generation (RAG)
- When and how to use tools vs. parametric knowledge

### What You Can Build After This Level
✅ AI assistants that can search the web  
✅ Data analysis applications  
✅ Document Q&A systems  
✅ API-integrated chatbots  

### 4.1 Function Calling Fundamentals

Modern LLMs can decide when and how to use tools. Here's how to implement this:

```python
import json
from typing import Dict, Any, List, Callable

class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict):
        """Register a tool with the AI system"""
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters
        }
    
    def get_tool_definitions(self):
        """Get tool definitions for the LLM"""
        definitions = []
        for name, tool in self.tools.items():
            definitions.append({
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            })
        return definitions
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]):
        """Execute a tool with given arguments"""
        if name not in self.tools:
            return f"Tool '{name}' not found"
        
        try:
            function = self.tools[name]["function"]
            return function(**arguments)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

# Example tools
def calculator(expression: str) -> str:
    """Perform mathematical calculations"""
    try:
        result = eval(expression)  # Use safer alternatives in production
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information"""
    # Mock implementation - replace with actual search API
    return f"Search results for '{query}': [Mock results 1-{num_results}]"

def get_weather(location: str) -> str:
    """Get current weather for a location"""
    # Mock implementation - replace with actual weather API
    return f"Weather in {location}: 72°F, sunny"

# Set up tool registry
tools = ToolRegistry()

tools.register_tool(
    "calculator",
    calculator,
    "Perform mathematical calculations",
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)

tools.register_tool(
    "web_search",
    web_search,
    "Search the web for current information",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
)

tools.register_tool(
    "get_weather",
    get_weather,
    "Get current weather information",
    {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or location name"
            }
        },
        "required": ["location"]
    }
)
```

### 4.2 Tool-Enabled AI Assistant

```python
class ToolEnabledAssistant:
    def __init__(self, tool_registry: ToolRegistry):
        self.tools = tool_registry
        
    def process_request(self, user_input: str) -> str:
        """Process user request, using tools when necessary"""
        
        # Create system prompt with tool information
        tool_descriptions = self.tools.get_tool_definitions()
        
        system_prompt = f"""
        You are a helpful assistant with access to the following tools:
        
        {json.dumps(tool_descriptions, indent=2)}
        
        When you need to use a tool:
        1. Determine which tool is most appropriate
        2. Extract the necessary parameters from the user's request
        3. Call the tool using this format:
        TOOL_CALL: {{"tool": "tool_name", "arguments": {{"param": "value"}}}}
        
        If no tool is needed, respond directly.
        """
        
        # Generate initial response
        prompt = f"""
        {system_prompt}
        
        User: {user_input}
        Assistant: """
        
        response = llm.generate(prompt)
        
        # Check if response contains tool calls
        if "TOOL_CALL:" in response:
            return self._process_tool_calls(response, user_input)
        else:
            return response
    
    def _process_tool_calls(self, response: str, original_input: str) -> str:
        """Process tool calls in the response"""
        parts = response.split("TOOL_CALL:")
        final_response = parts[0].strip()
        
        for part in parts[1:]:
            try:
                # Extract JSON from the tool call
                json_start = part.find("{")
                json_end = part.rfind("}") + 1
                tool_call_json = part[json_start:json_end]
                
                tool_call = json.loads(tool_call_json)
                tool_name = tool_call["tool"]
                arguments = tool_call["arguments"]
                
                # Execute the tool
                tool_result = self.tools.execute_tool(tool_name, arguments)
                
                # Generate follow-up response with tool result
                follow_up_prompt = f"""
                Original request: {original_input}
                Tool used: {tool_name}
                Tool result: {tool_result}
                
                Based on the tool result, provide a helpful response to the user:
                """
                
                follow_up_response = llm.generate(follow_up_prompt)
                final_response += f"\n\n{follow_up_response}"
                
            except Exception as e:
                final_response += f"\n\nError processing tool call: {str(e)}"
        
        return final_response

# Usage example
assistant = ToolEnabledAssistant(tools)

# Test the assistant
print(assistant.process_request("What's 15% of 2847?"))
print(assistant.process_request("What's the weather like in Tokyo?"))
print(assistant.process_request("Search for recent news about AI"))
```

### 4.3 Basic RAG Implementation

```python
import numpy as np
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer

class SimpleRAGSystem:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        
    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Add a document to the knowledge base"""
        # Split document into chunks
        chunks = self._chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedder.encode(chunk).tolist()
            
            self.collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{**(metadata or {}), "doc_id": doc_id, "chunk_index": i}]
            )
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [
            {
                "text": doc,
                "metadata": meta,
                "relevance_score": 1.0  # ChromaDB doesn't return scores by default
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    
    def answer_question(self, question: str, context_limit: int = 3000) -> str:
        """Answer question using retrieved context"""
        # Search for relevant documents
        relevant_docs = self.search(question, top_k=5)
        
        # Build context
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            doc_length = len(doc['text'])
            if current_length + doc_length <= context_limit:
                context_parts.append(doc['text'])
                current_length += doc_length
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""
        Context information:
        {context}
        
        Question: {question}
        
        Based on the context provided, please answer the question. If the context doesn't contain enough information to answer the question, please say so and explain what additional information would be needed.
        
        Answer:
        """
        
        answer = llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": [doc['metadata'] for doc in relevant_docs[:3]],
            "context_used": len(context_parts)
        }

# Usage example
rag = SimpleRAGSystem()

# Add some documents
rag.add_document(
    "ai_overview", 
    "Artificial Intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, perception, and language understanding.",
    {"category": "technology", "date": "2024"}
)

rag.add_document(
    "machine_learning",
    "Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience without being explicitly programmed. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
    {"category": "technology", "date": "2024"}
)

# Ask questions
result = rag.answer_question("What is the difference between AI and Machine Learning?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### 4.4 Choosing When to Use Tools vs. Parametric Knowledge

```python
class SmartAssistant:
    def __init__(self, tool_registry: ToolRegistry, rag_system: SimpleRAGSystem):
        self.tools = tool_registry
        self.rag = rag_system
        
    def decide_information_source(self, query: str) -> str:
        """Decide whether to use parametric knowledge, tools, or RAG"""
        
        analysis_prompt = f"""
        Analyze this query and determine the best information source:
        
        Query: {query}
        
        Consider:
        1. Does this require real-time/current information? (Use tools)
        2. Does this require specific factual lookup from documents? (Use RAG)
        3. Is this general knowledge that an AI would know? (Use parametric knowledge)
        4. Does this require computation or external API calls? (Use tools)
        
        Respond with one of: "parametric", "tools", "rag", "hybrid"
        Also provide reasoning.
        
        Format:
        Source: [parametric|tools|rag|hybrid]
        Reasoning: [explanation]
        """
        
        decision = llm.generate(analysis_prompt)
        return decision
    
    def process_query(self, query: str) -> str:
        """Process query using the most appropriate information source"""
        
        # Decide on information source
        decision = self.decide_information_source(query)
        
        if "parametric" in decision.lower():
            # Use the LLM's training knowledge
            response = llm.generate(f"Answer this question: {query}")
            
        elif "tools" in decision.lower():
            # Use external tools
            assistant = ToolEnabledAssistant(self.tools)
            response = assistant.process_request(query)
            
        elif "rag" in decision.lower():
            # Use RAG system
            result = self.rag.answer_question(query)
            response = result["answer"]
            
        elif "hybrid" in decision.lower():
            # Combine multiple sources
            response = self._hybrid_response(query)
            
        else:
            # Default to parametric
            response = llm.generate(f"Answer this question: {query}")
        
        return f"Decision process: {decision}\n\nAnswer: {response}"
    
    def _hybrid_response(self, query: str) -> str:
        """Generate response using multiple information sources"""
        
        # Get RAG results
        rag_result = self.rag.answer_question(query)
        
        # Get tool-based information if needed
        tool_assistant = ToolEnabledAssistant(self.tools)
        tool_response = tool_assistant.process_request(query)
        
        # Synthesize responses
        synthesis_prompt = f"""
        Query: {query}
        
        Information from knowledge base:
        {rag_result['answer']}
        
        Information from tools/current sources:
        {tool_response}
        
        Synthesize this information into a comprehensive, accurate answer:
        """
        
        return llm.generate(synthesis_prompt)

# Usage
smart_assistant = SmartAssistant(tools, rag)
print(smart_assistant.process_query("What is machine learning?"))  # Likely parametric/RAG
print(smart_assistant.process_query("What's the current weather in NYC?"))  # Tools
print(smart_assistant.process_query("Calculate the compound interest on $1000 at 5% for 10 years"))  # Tools
```

---

## Level 5: Advanced RAG & Knowledge Systems

### What You'll Learn
- Multi-hop reasoning with retrieval
- Hierarchical and hybrid RAG architectures
- Query understanding and decomposition
- Self-querying and adaptive retrieval

### What You Can Build After This Level
✅ Sophisticated knowledge management systems  
✅ Research assistants that can follow complex reasoning chains  
✅ Adaptive learning systems  
✅ Multi-modal knowledge bases  

### 5.1 Query Understanding and Decomposition

```python
class QueryAnalyzer:
    def __init__(self):
        pass
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and structure"""
        
        analysis_prompt = f"""
        Analyze this query in detail:
        
        Query: {query}
        
        Provide analysis in JSON format:
        {{
            "query_type": "factual|analytical|comparative|procedural|creative",
            "complexity": "simple|moderate|complex",
            "information_needs": ["list of information required"],
            "temporal_aspect": "historical|current|future|timeless",
            "scope": "narrow|broad",
            "sub_questions": ["break down into sub-questions if complex"],
            "key_concepts": ["main concepts to search for"],
            "filters": {{
                "date_range": "if time-specific",
                "category": "if domain-specific",
                "source_type": "if source preference indicated"
            }}
        }}
        """
        
        analysis = llm.generate(analysis_prompt)
        return json.loads(analysis)
    
    def decompose_complex_query(self, query: str) -> List[Dict[str, Any]]:
        """Break complex queries into simpler sub-queries"""
        
        analysis = self.analyze_query(query)
        
        if analysis["complexity"] == "simple":
            return [{"query": query, "dependency": None, "type": "direct"}]
        
        decomposition_prompt = f"""
        Break down this complex query into simpler sub-queries:
        
        Original query: {query}
        Query analysis: {analysis}
        
        Create a query plan as JSON:
        {{
            "sub_queries": [
                {{
                    "id": "query_1",
                    "question": "first sub-question",
                    "dependencies": [],
                    "search_type": "semantic|keyword|hybrid",
                    "expected_answer_type": "factual|list|explanation"
                }},
                {{
                    "id": "query_2", 
                    "question": "second sub-question that might depend on query_1",
                    "dependencies": ["query_1"],
                    "search_type": "semantic|keyword|hybrid",
                    "expected_answer_type": "factual|list|explanation"
                }}
            ],
            "synthesis_strategy": "how to combine the sub-query results"
        }}
        """
        
        decomposition = llm.generate(decomposition_prompt)
        return json.loads(decomposition)

class AdvancedRAGSystem:
    def __init__(self):
        self.query_analyzer = QueryAnalyzer()
        self.basic_rag = SimpleRAGSystem()
        self.search_history = []
        
    def multi_hop_search(self, query: str) -> Dict[str, Any]:
        """Perform multi-hop reasoning with retrieval"""
        
        # Analyze and decompose query
        query_plan = self.query_analyzer.decompose_complex_query(query)
        
        if isinstance(query_plan, dict) and "sub_queries" in query_plan:
            return self._execute_query_plan(query_plan)
        else:
            # Simple query, use basic RAG
            return self.basic_rag.answer_question(query)
    
    def _execute_query_plan(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complex query plan with dependencies"""
        
        sub_results = {}
        execution_order = self._topological_sort(query_plan["sub_queries"])
        
        for query_id in execution_order:
            sub_query = next(q for q in query_plan["sub_queries"] if q["id"] == query_id)
            
            # Build context from dependencies
            dependency_context = ""
            for dep_id in sub_query.get("dependencies", []):
                if dep_id in sub_results:
                    dependency_context += f"\nPrevious finding ({dep_id}): {sub_results[dep_id]['answer']}"
            
            # Execute sub-query with context
            if dependency_context:
                enhanced_query = f"{sub_query['question']}\n\nContext from previous steps:{dependency_context}"
            else:
                enhanced_query = sub_query['question']
            
            result = self.basic_rag.answer_question(enhanced_query)
            sub_results[query_id] = result
        
        # Synthesize final answer
        synthesis_prompt = f"""
        Original question: {query_plan.get('original_query', 'Complex query')}
        
        Sub-query results:
        {json.dumps({k: v['answer'] for k, v in sub_results.items()}, indent=2)}
        
        Synthesis strategy: {query_plan.get('synthesis_strategy', 'Combine all findings')}
        
        Provide a comprehensive answer that synthesizes all the sub-query results:
        """
        
        final_answer = llm.generate(synthesis_prompt)
        
        return {
            "answer": final_answer,
            "sub_results": sub_results,
            "reasoning_chain": execution_order
        }
    
    def _topological_sort(self, sub_queries: List[Dict]) -> List[str]:
        """Sort sub-queries by dependencies"""
        # Simple topological sort implementation
        in_degree = {q["id"]: len(q.get("dependencies", [])) for q in sub_queries}
        queue = [q_id for q_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update dependencies
            for query in sub_queries:
                if current in query.get("dependencies", []):
                    in_degree[query["id"]] -= 1
                    if in_degree[query["id"]] == 0:
                        queue.append(query["id"])
        
        return result
```

### 5.2 Hierarchical RAG (HRAG)

Multi-tiered retrieval systems that organize information in hierarchical structures.

```python
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

class HierarchicalRAG:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        
        # Multiple index levels
        self.document_index = None  # Document summaries
        self.section_index = None   # Section-level content
        self.chunk_index = None     # Detailed chunks
        
    def build_hierarchical_index(self, documents):
        """Build multi-level index hierarchy"""
        
        # Level 1: Document summaries
        summaries = self.create_document_summaries(documents)
        self.document_index = FAISS.from_texts(summaries, self.embeddings)
        
        # Level 2: Section-level content
        sections = self.extract_sections(documents)
        self.section_index = FAISS.from_texts(sections, self.embeddings)
        
        # Level 3: Detailed chunks
        chunks = self.create_detailed_chunks(documents)
        self.chunk_index = FAISS.from_texts(chunks, self.embeddings)
    
    def create_document_summaries(self, documents):
        """Create high-level summaries for document-level index"""
        summaries = []
        
        for doc in documents:
            summary_prompt = f"""
            Create a comprehensive summary of this document that captures:
            - Main topics and themes
            - Key concepts and entities
            - Important conclusions or findings
            - Document purpose and scope
            
            Document: {doc.page_content[:2000]}...
            
            Summary:
            """
            
            summary = self.llm.generate(summary_prompt)
            summaries.append(summary)
        
        return summaries
    
    def hierarchical_retrieve(self, query: str, k_docs=3, k_sections=5, k_chunks=10):
        """Perform hierarchical retrieval"""
        
        # Step 1: Find relevant documents
        relevant_docs = self.document_index.similarity_search(query, k=k_docs)
        doc_ids = [doc.metadata.get('doc_id') for doc in relevant_docs if 'doc_id' in doc.metadata]
        
        # Step 2: Find relevant sections within those documents
        if doc_ids:
            section_filter = {"doc_id": {"$in": doc_ids}}
            relevant_sections = self.section_index.similarity_search(
                query, k=k_sections, filter=section_filter
            )
        else:
            relevant_sections = self.section_index.similarity_search(query, k=k_sections)
        
        # Step 3: Find detailed chunks within relevant sections
        section_ids = [sec.metadata.get('section_id') for sec in relevant_sections if 'section_id' in sec.metadata]
        
        if section_ids:
            chunk_filter = {"section_id": {"$in": section_ids}}
            final_chunks = self.chunk_index.similarity_search(
                query, k=k_chunks, filter=chunk_filter
            )
        else:
            final_chunks = self.chunk_index.similarity_search(query, k=k_chunks)
        
        return {
            "documents": relevant_docs,
            "sections": relevant_sections,
            "chunks": final_chunks
        }
    
    def answer_with_hierarchy(self, query: str):
        """Generate answer using hierarchical context"""
        
        retrieval_results = self.hierarchical_retrieve(query)
        
        # Build hierarchical context
        context_parts = [
            "=== DOCUMENT OVERVIEW ===",
            "\n".join([doc.page_content for doc in retrieval_results["documents"][:2]]),
            "\n=== RELEVANT SECTIONS ===", 
            "\n".join([sec.page_content for sec in retrieval_results["sections"][:3]]),
            "\n=== DETAILED INFORMATION ===",
            "\n".join([chunk.page_content for chunk in retrieval_results["chunks"][:5]])
        ]
        
        hierarchical_context = "\n".join(context_parts)
        
        answer_prompt = f"""
        Question: {query}
        
        Context (organized from general to specific):
        {hierarchical_context}
        
        Instructions:
        - Start with the big picture from document overviews
        - Use section content for mid-level understanding
        - Reference specific details from detailed information
        - Synthesize across all levels for a comprehensive answer
        
        Answer:
        """
        
        return self.llm.generate(answer_prompt)

### 5.3 Graph RAG

Integrating knowledge graphs with RAG for enhanced relationship-based retrieval.

```python
from neo4j import GraphDatabase
from langchain.graphs import Neo4jGraph
import spacy

class GraphRAG:
    def __init__(self, neo4j_uri, username, password, llm):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=username,
            password=password
        )
        self.llm = llm
        self.nlp = spacy.load("en_core_web_sm")  # For entity extraction
        
    def extract_entities(self, text: str) -> list:
        """Extract entities from text using NLP"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities
    
    def graph_enhanced_retrieval(self, query: str):
        """Retrieve using both graph relationships and vector similarity"""
        
        # Extract entities from query
        query_entities = self.extract_entities(query)
        entity_names = [ent["text"] for ent in query_entities]
        
        # Get graph context for entities
        graph_context = self.get_graph_context(entity_names)
        
        # Combine with traditional vector search
        vector_results = self.vector_search(query)
        
        # Merge and rank results
        combined_results = self.merge_graph_vector_results(graph_context, vector_results)
        
        return combined_results
    
    def get_graph_context(self, entities: list, max_hops: int = 2):
        """Retrieve graph context for entities"""
        
        if not entities:
            return []
        
        # Cypher query to find relationships
        cypher_query = f"""
        MATCH (e:Entity)-[r*1..{max_hops}]-(related:Entity)
        WHERE e.name IN $entities
        RETURN e.name as entity, 
               collect(DISTINCT related.name) as related_entities,
               collect(DISTINCT type(r)) as relationship_types
        LIMIT 100
        """
        
        try:
            results = self.graph.query(cypher_query, {"entities": entities})
            return self.format_graph_results(results)
        except Exception as e:
            print(f"Graph query error: {e}")
            return []
    
    def format_graph_results(self, graph_results):
        """Format graph query results for context"""
        
        formatted_results = []
        
        for result in graph_results:
            entity = result["entity"]
            related = result["related_entities"][:10]  # Limit related entities
            relationships = result["relationship_types"]
            
            context_text = f"""
            Entity: {entity}
            Related to: {', '.join(related)}
            Relationship types: {', '.join(set(relationships))}
            """
            
            formatted_results.append(context_text.strip())
        
        return formatted_results
    
    def merge_graph_vector_results(self, graph_context, vector_results):
        """Intelligently merge graph and vector search results"""
        
        # Enhance vector results with graph context
        enhanced_results = []
        
        for vector_result in vector_results:
            enhanced_result = {
                "content": vector_result.page_content,
                "metadata": vector_result.metadata,
                "graph_context": [],
                "relevance_score": getattr(vector_result, 'score', 0.5)
            }
            
            # Find relevant graph context
            for graph_item in graph_context:
                if self.has_entity_overlap(vector_result.page_content, graph_item):
                    enhanced_result["graph_context"].append(graph_item)
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def has_entity_overlap(self, text1: str, text2: str) -> bool:
        """Check if two texts share entities"""
        entities1 = set([ent["text"].lower() for ent in self.extract_entities(text1)])
        entities2 = set([ent["text"].lower() for ent in self.extract_entities(text2)])
        
        return len(entities1.intersection(entities2)) > 0

### 5.4 Multimodal RAG

Advanced systems handling text, images, tables, and other data types.

```python
from langchain.document_loaders import PyMuPDFLoader
import pytesseract
from PIL import Image
import pandas as pd

class MultimodalRAG:
    def __init__(self, llm, text_embeddings):
        self.llm = llm
        self.text_embeddings = text_embeddings
        # Note: In production, you'd use specialized embeddings for each modality
        self.image_processor = self.setup_image_processor()
        self.table_processor = self.setup_table_processor()
        
        # Separate vector stores for each modality
        self.text_store = None
        self.image_store = None
        self.table_store = None
        
    def setup_image_processor(self):
        """Setup image processing capabilities"""
        def process_image(image_path):
            try:
                # Extract text from image using OCR
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                
                # Describe image content (simplified)
                description = self.describe_image(image)
                
                return {
                    "ocr_text": text,
                    "description": description,
                    "image_path": image_path
                }
            except Exception as e:
                return {"error": str(e)}
        
        return process_image
    
    def describe_image(self, image):
        """Generate description of image content"""
        # In production, use vision-language models like CLIP or BLIP
        # For now, return placeholder
        return "Image description would be generated by vision model"
    
    def setup_table_processor(self):
        """Setup table processing capabilities"""
        def process_table(table_data):
            if isinstance(table_data, str):
                # Parse table from string/HTML
                df = pd.read_html(table_data)[0]
            elif isinstance(table_data, pd.DataFrame):
                df = table_data
            else:
                return {"error": "Unsupported table format"}
            
            # Create searchable representation
            table_summary = self.summarize_table(df)
            column_info = self.analyze_columns(df)
            
            return {
                "summary": table_summary,
                "columns": column_info,
                "row_count": len(df),
                "searchable_content": self.create_table_search_content(df)
            }
        
        return process_table
    
    def process_multimodal_document(self, document_path):
        """Extract and process multimodal content"""
        
        loader = PyMuPDFLoader(document_path)
        pages = loader.load()
        
        extracted_content = {
            "text": [],
            "images": [],
            "tables": [],
            "metadata": {"source": document_path}
        }
        
        for page_num, page in enumerate(pages):
            # Extract text
            if page.page_content.strip():
                extracted_content["text"].append({
                    "content": page.page_content,
                    "page": page_num,
                    "type": "text"
                })
            
            # Extract images (would need additional implementation)
            page_images = self.extract_images_from_page(page)
            for img in page_images:
                processed_img = self.image_processor(img["path"])
                extracted_content["images"].append({
                    **processed_img,
                    "page": page_num,
                    "type": "image"
                })
            
            # Extract tables (would need additional implementation)
            page_tables = self.extract_tables_from_page(page)
            for table in page_tables:
                processed_table = self.table_processor(table)
                extracted_content["tables"].append({
                    **processed_table,
                    "page": page_num,
                    "type": "table"
                })
        
        return extracted_content
    
    def multimodal_search(self, query: str, modalities=None):
        """Search across multiple modalities"""
        
        if modalities is None:
            modalities = ["text", "image", "table"]
        
        results = {"text": [], "images": [], "tables": []}
        
        if "text" in modalities and self.text_store:
            results["text"] = self.text_store.similarity_search(query, k=5)
        
        if "image" in modalities and self.image_store:
            # Search image descriptions and OCR text
            results["images"] = self.search_images(query)
        
        if "table" in modalities and self.table_store:
            # Search table summaries and content
            results["tables"] = self.search_tables(query)
        
        return self.fuse_multimodal_results(results, query)
    
    def fuse_multimodal_results(self, results: dict, query: str):
        """Intelligently combine results from different modalities"""
        
        fused_results = []
        
        # Score and combine results from each modality
        for modality, modality_results in results.items():
            for result in modality_results:
                fused_result = {
                    "content": self.format_content_for_modality(result, modality),
                    "modality": modality,
                    "relevance_score": self.calculate_cross_modal_relevance(result, query),
                    "metadata": getattr(result, 'metadata', {})
                }
                fused_results.append(fused_result)
        
        # Sort by relevance score
        fused_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return fused_results[:10]  # Return top 10 cross-modal results
    
    def generate_multimodal_answer(self, query: str, fused_results: list):
        """Generate answer using multimodal context"""
        
        # Organize context by modality
        context_parts = []
        
        text_content = [r for r in fused_results if r["modality"] == "text"]
        if text_content:
            context_parts.append("TEXT INFORMATION:")
            context_parts.extend([r["content"] for r in text_content[:3]])
        
        image_content = [r for r in fused_results if r["modality"] == "image"]
        if image_content:
            context_parts.append("\nIMAGE INFORMATION:")
            context_parts.extend([r["content"] for r in image_content[:2]])
        
        table_content = [r for r in fused_results if r["modality"] == "table"]
        if table_content:
            context_parts.append("\nTABULAR DATA:")
            context_parts.extend([r["content"] for r in table_content[:2]])
        
        multimodal_context = "\n".join(context_parts)
        
        answer_prompt = f"""
        Question: {query}
        
        Multimodal Context:
        {multimodal_context}
        
        Instructions:
        - Synthesize information from text, images, and tables
        - Explicitly reference the source type when relevant
        - Provide a comprehensive answer that leverages all available modalities
        - Note if visual or tabular information provides unique insights
        
        Answer:
        """
        
        return self.llm.generate(answer_prompt)

### 5.5 Agentic RAG Systems

Self-reflective and adaptive RAG systems that improve their retrieval strategies.

```python
class AgenticRAG:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.critic = self.create_critic_agent()
        self.query_refiner = self.create_query_refiner()
        
    def create_critic_agent(self):
        """Create a critic agent for self-reflection"""
        
        def critique_response(query, response, context):
            critique_prompt = f"""
            Evaluate this Q&A interaction:
            
            Question: {query}
            Generated Answer: {response}
            Context Used: {context[:500]}...
            
            Evaluation criteria:
            1. Relevance: Does the answer address the question?
            2. Accuracy: Is the information correct based on context?
            3. Completeness: Does it fully answer the question?
            4. Context Usage: Was the context used effectively?
            
            Provide scores (1-10) and specific feedback:
            
            Scores:
            - Relevance: X/10
            - Accuracy: X/10  
            - Completeness: X/10
            - Context Usage: X/10
            
            Feedback: [Specific suggestions for improvement]
            
            Needs Improvement: Yes/No
            """
            
            return self.llm.generate(critique_prompt)
        
        return critique_response
    
    def create_query_refiner(self):
        """Create query refinement agent"""
        
        def refine_query(original_query, critique_feedback):
            refinement_prompt = f"""
            Original Query: {original_query}
            Critique Feedback: {critique_feedback}
            
            Based on the feedback, create an improved search query that will:
            1. Retrieve more relevant information
            2. Address any gaps identified in the critique
            3. Be more specific or broader as needed
            
            Refined Query: 
            Reasoning: [Explain the refinements made]
            """
            
            return self.llm.generate(refinement_prompt)
        
        return refine_query
    
    def self_reflective_retrieval(self, query: str, max_iterations: int = 3):
        """Perform retrieval with self-reflection and improvement"""
        
        current_query = query
        iteration_results = []
        
        for iteration in range(max_iterations):
            # Retrieve documents
            docs = self.retriever.get_relevant_documents(current_query)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Generate response
            response = self.generate_response(current_query, context)
            
            # Self-critique
            critique = self.critic(current_query, response, context)
            
            iteration_result = {
                "iteration": iteration + 1,
                "query": current_query,
                "response": response,
                "critique": critique,
                "documents_retrieved": len(docs)
            }
            iteration_results.append(iteration_result)
            
            # Check if improvement is needed
            if "Needs Improvement: No" in critique or iteration == max_iterations - 1:
                break
            
            # Refine query for next iteration
            refinement = self.query_refiner(current_query, critique)
            current_query = self.extract_refined_query(refinement)
        
        return {
            "final_response": iteration_results[-1]["response"],
            "iterations": iteration_results,
            "query_evolution": [r["query"] for r in iteration_results]
        }
    
    def corrective_rag(self, query: str, relevance_threshold: float = 0.6):
        """Implementation of Corrective RAG (CRAG)"""
        
        # Initial retrieval
        docs = self.retriever.get_relevant_documents(query)
        
        # Grade document relevance
        relevance_scores = []
        for doc in docs:
            score = self.grade_document_relevance(doc, query)
            relevance_scores.append(score)
        
        # Filter based on relevance
        relevant_docs = [
            doc for doc, score in zip(docs, relevance_scores)
            if score >= relevance_threshold
        ]
        
        # If no relevant documents, fall back to web search
        if not relevant_docs:
            web_results = self.web_search_fallback(query)
            relevant_docs.extend(web_results)
        
        # If some documents are borderline, refine them
        refined_docs = []
        for doc, score in zip(docs, relevance_scores):
            if score >= relevance_threshold:
                refined_docs.append(doc)
            elif score >= relevance_threshold * 0.7:  # Borderline
                refined_doc = self.refine_document(doc, query)
                refined_docs.append(refined_doc)
        
        # Generate final response
        context = "\n".join([doc.page_content for doc in refined_docs])
        response = self.generate_response(query, context)
        
        return {
            "response": response,
            "original_docs": len(docs),
            "relevant_docs": len(relevant_docs),
            "relevance_scores": relevance_scores,
            "used_web_fallback": len(relevant_docs) == 0
        }
    
    def grade_document_relevance(self, document, query: str) -> float:
        """Grade how relevant a document is to the query"""
        
        grading_prompt = f"""
        Query: {query}
        Document: {document.page_content[:500]}...
        
        Rate the relevance of this document to the query on a scale of 0.0 to 1.0:
        - 1.0: Highly relevant, directly answers the query
        - 0.7-0.9: Relevant, contains useful information
        - 0.4-0.6: Somewhat relevant, tangentially related
        - 0.0-0.3: Not relevant, off-topic
        
        Consider:
        - Topical overlap
        - Information quality
        - Specificity to the query
        
        Relevance Score: [0.0-1.0]
        Brief Reasoning: [Explain the score]
        """
        
        grading_result = self.llm.generate(grading_prompt)
        
        # Extract numerical score
        try:
            score_line = [line for line in grading_result.split('\n') if 'Score:' in line][0]
            score = float(score_line.split(':')[1].strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except:
            return 0.5  # Default score if parsing fails
```

### 5.3 Hierarchical RAG

```python
class HierarchicalRAGSystem:
    def __init__(self):
        # Different levels of the hierarchy
        self.document_level_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_level_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Separate collections for different levels
        self.client = chromadb.Client()
        self.document_collection = self.client.create_collection("documents")
        self.chunk_collection = self.client.create_collection("chunks")
        self.paragraph_collection = self.client.create_collection("paragraphs")
    
    def add_document_hierarchical(self, doc_id: str, title: str, content: str, metadata: Dict = None):
        """Add document with hierarchical indexing"""
        
        # Document-level summary
        summary = self._create_document_summary(title, content)
        doc_embedding = self.document_level_embedder.encode(summary).tolist()
        
        self.document_collection.add(
            ids=[doc_id],
            embeddings=[doc_embedding],
            documents=[summary],
            metadatas=[{**(metadata or {}), "title": title, "doc_type": "summary"}]
        )
        
        # Paragraph-level indexing
        paragraphs = content.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 50:  # Only index substantial paragraphs
                para_id = f"{doc_id}_para_{i}"
                para_embedding = self.chunk_level_embedder.encode(paragraph).tolist()
                
                self.paragraph_collection.add(
                    ids=[para_id],
                    embeddings=[para_embedding],
                    documents=[paragraph],
                    metadatas=[{**(metadata or {}), "doc_id": doc_id, "paragraph_index": i}]
                )
        
        # Chunk-level indexing (smaller pieces)
        chunks = self._create_chunks(content, chunk_size=300)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_embedding = self.chunk_level_embedder.encode(chunk).tolist()
            
            self.chunk_collection.add(
                ids=[chunk_id],
                embeddings=[chunk_embedding], 
                documents=[chunk],
                metadatas=[{**(metadata or {}), "doc_id": doc_id, "chunk_index": i}]
            )
    
    def hierarchical_search(self, query: str, max_docs: int = 5) -> Dict[str, Any]:
        """Perform hierarchical search from documents down to chunks"""
        
        # Level 1: Find relevant documents
        query_embedding = self.document_level_embedder.encode(query).tolist()
        doc_results = self.document_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_docs
        )
        
        relevant_doc_ids = [
            meta["doc_id"] if "doc_id" in meta else doc_id 
            for meta, doc_id in zip(doc_results['metadatas'][0], doc_results['ids'][0])
        ]
        
        # Level 2: Find relevant paragraphs within those documents
        paragraph_results = []
        for doc_id in relevant_doc_ids:
            para_query = self.paragraph_collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"doc_id": doc_id}
            )
            paragraph_results.extend(list(zip(
                para_query['documents'][0],
                para_query['metadatas'][0]
            )))
        
        # Level 3: Find specific chunks within relevant paragraphs
        chunk_results = []
        for doc_id in relevant_doc_ids:
            chunk_query = self.chunk_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"doc_id": doc_id}
            )
            chunk_results.extend(list(zip(
                chunk_query['documents'][0],
                chunk_query['metadatas'][0]
            )))
        
        return {
            "document_level": list(zip(doc_results['documents'][0], doc_results['metadatas'][0])),
            "paragraph_level": paragraph_results,
            "chunk_level": chunk_results
        }
    
    def _create_document_summary(self, title: str, content: str, max_length: int = 500) -> str:
        """Create a summary of the document for document-level indexing"""
        
        summary_prompt = f"""
        Create a comprehensive summary of this document:
        
        Title: {title}
        Content: {content[:2000]}...  # Truncate for prompt
        
        Summary should:
        - Capture main topics and themes
        - Include key concepts and entities
        - Be suitable for semantic search
        - Be around {max_length} characters
        
        Summary:
        """
        
        return llm.generate(summary_prompt)
    
    def answer_with_hierarchy(self, query: str) -> Dict[str, Any]:
        """Answer question using hierarchical context"""
        
        # Get hierarchical search results
        search_results = self.hierarchical_search(query)
        
        # Build context from different levels
        context_parts = []
        
        # Add document-level context for broad understanding
        context_parts.append("Document overviews:")
        for doc, meta in search_results["document_level"][:2]:
            context_parts.append(f"- {meta.get('title', 'Unknown')}: {doc}")
        
        # Add paragraph-level context for mid-level detail
        context_parts.append("\nRelevant sections:")
        for para, meta in search_results["paragraph_level"][:3]:
            context_parts.append(f"- {para}")
        
        # Add chunk-level context for specific details
        context_parts.append("\nSpecific details:")
        for chunk, meta in search_results["chunk_level"][:5]:
            context_parts.append(f"- {chunk}")
        
        full_context = "\n".join(context_parts)
        
        # Generate answer
        answer_prompt = f"""
        Question: {query}
        
        Context (from broad to specific):
        {full_context}
        
        Provide a comprehensive answer using the hierarchical context. Start with the big picture and then provide specific details.
        
        Answer:
        """
        
        answer = llm.generate(answer_prompt)
        
        return {
            "answer": answer,
            "context_sources": {
                "documents": len(search_results["document_level"]),
                "paragraphs": len(search_results["paragraph_level"]),
                "chunks": len(search_results["chunk_level"])
            }
        }
```

### 5.4 Adaptive RAG with Feedback

```python
class AdaptiveRAGSystem:
    def __init__(self):
        self.rag = AdvancedRAGSystem()
        self.query_performance = {}  # Track query performance
        self.search_strategies = ["semantic", "keyword", "hybrid", "hierarchical"]
        
    def answer_with_adaptation(self, query: str, user_feedback: str = None) -> Dict[str, Any]:
        """Answer query with adaptive strategy based on performance"""
        
        # Choose initial strategy based on query type and history
        strategy = self._choose_strategy(query)
        
        # Attempt answer with chosen strategy
        result = self._execute_strategy(query, strategy)
        
        # Evaluate result quality
        quality_score = self._evaluate_result_quality(query, result)
        
        # If quality is poor, try alternative strategy
        if quality_score < 0.7 and strategy != "hybrid":
            alternative_strategy = self._get_alternative_strategy(strategy)
            alternative_result = self._execute_strategy(query, alternative_strategy)
            alternative_quality = self._evaluate_result_quality(query, alternative_result)
            
            if alternative_quality > quality_score:
                result = alternative_result
                strategy = alternative_strategy
                quality_score = alternative_quality
        
        # Record performance for future adaptation
        self._record_performance(query, strategy, quality_score, user_feedback)
        
        return {
            **result,
            "strategy_used": strategy,
            "quality_score": quality_score,
            "adapted": quality_score < 0.7
        }
    
    def _choose_strategy(self, query: str) -> str:
        """Choose search strategy based on query characteristics and history"""
        
        # Analyze query characteristics
        query_analysis = self.rag.query_analyzer.analyze_query(query)
        
        # Simple strategy selection logic (could be ML-based)
        if query_analysis["query_type"] == "factual" and query_analysis["complexity"] == "simple":
            return "semantic"
        elif query_analysis["query_type"] == "analytical":
            return "hierarchical"
        elif "recent" in query.lower() or "latest" in query.lower():
            return "keyword"
        else:
            return "hybrid"
    
    def _execute_strategy(self, query: str, strategy: str) -> Dict[str, Any]:
        """Execute specific search strategy"""
        
        if strategy == "semantic":
            return self.rag.basic_rag.answer_question(query)
        elif strategy == "hierarchical":
            hierarchical_rag = HierarchicalRAGSystem()
            return hierarchical_rag.answer_with_hierarchy(query)
        elif strategy == "keyword":
            # Implement keyword-based search
            return self._keyword_search(query)
        elif strategy == "hybrid":
            return self.rag.multi_hop_search(query)
        else:
            return self.rag.basic_rag.answer_question(query)
    
    def _evaluate_result_quality(self, query: str, result: Dict[str, Any]) -> float:
        """Evaluate quality of search result"""
        
        evaluation_prompt = f"""
        Evaluate the quality of this answer to the given question:
        
        Question: {query}
        Answer: {result.get('answer', '')}
        
        Rate on a scale of 0.0 to 1.0 based on:
        - Relevance to the question
        - Completeness of the answer  
        - Accuracy (as far as can be determined)
        - Clarity and coherence
        
        Provide only the numerical score (e.g., 0.85):
        """
        
        score_text = llm.generate(evaluation_prompt)
        try:
            return float(score_text.strip())
        except:
            return 0.5  # Default middle score if parsing fails
    
    def _record_performance(self, query: str, strategy: str, quality_score: float, feedback: str = None):
        """Record performance for future strategy selection"""
        
        query_type = self.rag.query_analyzer.analyze_query(query)["query_type"]
        
        if query_type not in self.query_performance:
            self.query_performance[query_type] = {}
        
        if strategy not in self.query_performance[query_type]:
            self.query_performance[query_type][strategy] = []
        
        self.query_performance[query_type][strategy].append({
            "quality_score": quality_score,
            "feedback": feedback,
            "timestamp": time.time()
        })
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about strategy performance"""
        
        insights = {}
        
        for query_type, strategies in self.query_performance.items():
            insights[query_type] = {}
            
            for strategy, performances in strategies.items():
                scores = [p["quality_score"] for p in performances]
                insights[query_type][strategy] = {
                    "avg_score": sum(scores) / len(scores),
                    "total_queries": len(scores),
                    "trend": "improving" if len(scores) > 5 and scores[-3:] > scores[:3] else "stable"
                }
        
        return insights
```

---

## Level 6: Single Agent Workflows

### What You'll Learn
- Autonomous reasoning and planning
- Tool orchestration and workflow management
- Memory and state management for agents
- Error handling and recovery strategies

### What You Can Build After This Level
✅ Autonomous task completion systems  
✅ Research and analysis agents  
✅ Content creation workflows  
✅ Personal AI assistants with persistence  

### 6.1 Core Agent Architecture

```python
from typing import List, Dict, Any, Optional
import time
import uuid

class AutonomousAgent:
    def __init__(self, name: str, role: str, tools: List = None):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.tools = tools or []
        
        # Agent state
        self.working_memory = []
        self.long_term_memory = []
        self.current_task = None
        self.plan = None
        self.execution_state = "idle"  # idle, planning, executing, paused, completed, failed
        
        # Performance tracking
        self.completed_tasks = []
        self.failed_tasks = []
        
    def receive_task(self, task: str, context: Dict = None) -> str:
        """Receive and begin processing a new task"""
        
        self.current_task = {
            "id": str(uuid.uuid4()),
            "description": task,
            "context": context or {},
            "start_time": time.time(),
            "status": "received"
        }
        
        self.execution_state = "planning"
        
        # Add task to working memory
        self._add_to_memory("task_received", {
            "task": task,
            "context": context
        })
        
        return f"Task received: {task}. Beginning analysis and planning."
    
    async def execute_task(self) -> Dict[str, Any]:
        """Main execution loop for the agent"""
        
        if not self.current_task:
            return {"error": "No task to execute"}
        
        try:
            # Step 1: Create execution plan
            self.execution_state = "planning"
            plan = await self._create_plan()
            self.plan = plan
            
            # Step 2: Execute plan
            self.execution_state = "executing"
            results = await self._execute_plan(plan)
            
            # Step 3: Evaluate and finalize
            self.execution_state = "completed"
            final_result = await self._finalize_task(results)
            
            # Record success
            self.completed_tasks.append(self.current_task)
            
            return final_result
            
        except Exception as e:
            self.execution_state = "failed"
            self.failed_tasks.append({
                **self.current_task,
                "error": str(e),
                "failed_at": time.time()
            })
            
            return {
                "success": False,
                "error": str(e),
                "recovery_suggestions": await self._suggest_recovery(str(e))
            }
    
    async def _create_plan(self) -> Dict[str, Any]:
        """Create detailed execution plan for the current task"""
        
        # Analyze task requirements
        task_analysis = await self._analyze_task()
        
        # Create step-by-step plan
        planning_prompt = f"""
        Task: {self.current_task['description']}
        Context: {self.current_task.get('context', {})}
        Task Analysis: {task_analysis}
        
        Available tools: {[tool.name for tool in self.tools]}
        My role: {self.role}
        
        Create a detailed execution plan:
        
        {{
            "goal": "clear statement of what to achieve",
            "approach": "high-level strategy",
            "steps": [
                {{
                    "id": "step_1",
                    "description": "what to do in this step",
                    "tool_needed": "tool name or 'reasoning'",
                    "expected_output": "what this step should produce",
                    "dependencies": ["list of step ids that must complete first"],
                    "success_criteria": "how to know this step succeeded"
                }}
            ],
            "success_metrics": "how to measure overall success",
            "potential_risks": ["things that could go wrong"],
            "fallback_plans": ["alternative approaches if main plan fails"]
        }}
        
        Plan:
        """
        
        plan_response = await llm.generate(planning_prompt)
        plan = json.loads(plan_response)
        
        # Store plan in memory
        self._add_to_memory("plan_created", plan)
        
        return plan
    
    async def _analyze_task(self) -> Dict[str, Any]:
        """Analyze task to understand requirements and complexity"""
        
        analysis_prompt = f"""
        Analyze this task in detail:
        
        Task: {self.current_task['description']}
        Context: {self.current_task.get('context', {})}
        My role: {self.role}
        
        Provide analysis:
        {{
            "task_type": "research|analysis|creation|communication|problem_solving",
            "complexity": "low|medium|high",
            "estimated_duration": "time estimate",
            "required_capabilities": ["list of capabilities needed"],
            "information_needs": ["what information I need to gather"],
            "deliverable_type": "what type of output is expected",
            "constraints": ["any limitations or requirements"],
            "success_criteria": ["how to measure success"]
        }}
        """
        
        analysis_response = await llm.generate(analysis_prompt)
        return json.loads(analysis_response)
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the created plan step by step"""
        
        step_results = {}
        execution_log = []
        
        # Execute steps in dependency order
        execution_order = self._resolve_dependencies(plan["steps"])
        
        for step_id in execution_order:
            step = next(s for s in plan["steps"] if s["id"] == step_id)
            
            # Execute individual step
            step_start_time = time.time()
            try:
                step_result = await self._execute_step(step, step_results)
                step_results[step_id] = step_result
                
                execution_log.append({
                    "step_id": step_id,
                    "status": "completed",
                    "duration": time.time() - step_start_time,
                    "result_summary": step_result.get("summary", "")
                })
                
                # Add to working memory
                self._add_to_memory("step_completed", {
                    "step": step,
                    "result": step_result
                })
                
            except Exception as e:
                execution_log.append({
                    "step_id": step_id,
                    "status": "failed",
                    "error": str(e),
                    "duration": time.time() - step_start_time
                })
                
                # Try to recover or continue
                recovery_action = await self._handle_step_failure(step, str(e))
                if recovery_action["action"] == "abort":
                    raise Exception(f"Step {step_id} failed: {str(e)}")
                elif recovery_action["action"] == "retry":
                    # Retry logic would go here
                    pass
                elif recovery_action["action"] == "skip":
                    step_results[step_id] = {"status": "skipped", "reason": str(e)}
        
        return {
            "step_results": step_results,
            "execution_log": execution_log,
            "overall_status": "completed"
        }
    
    async def _execute_step(self, step: Dict[str, Any], previous_results: Dict) -> Dict[str, Any]:
        """Execute a single step of the plan"""
        
        tool_needed = step.get("tool_needed", "reasoning")
        
        if tool_needed == "reasoning":
            # Pure reasoning step
            reasoning_prompt = f"""
            Step: {step['description']}
            Expected output: {step['expected_output']}
            Previous results: {previous_results}
            Current context: {self._get_relevant_memory()}
            
            Perform this reasoning step and provide the expected output:
            """
            
            result = await llm.generate(reasoning_prompt)
            return {
                "type": "reasoning",
                "output": result,
                "summary": step['description']
            }
        
        else:
            # Tool-based step
            tool = self._find_tool(tool_needed)
            if not tool:
                raise Exception(f"Required tool '{tool_needed}' not available")
            
            # Determine tool parameters from step description and context
            parameters = await self._determine_tool_parameters(step, previous_results, tool)
            
            # Execute tool
            tool_result = await tool.execute(parameters)
            
            return {
                "type": "tool_execution",
                "tool": tool_needed,
                "parameters": parameters,
                "output": tool_result,
                "summary": f"Used {tool_needed}: {step['description']}"
            }
    
    async def _determine_tool_parameters(self, step: Dict, previous_results: Dict, tool) -> Dict:
        """Determine parameters for tool execution based on step context"""
        
        parameter_prompt = f"""
        I need to execute this tool: {tool.name}
        Tool description: {tool.description}
        Tool parameters schema: {tool.parameters}
        
        For this step: {step['description']}
        Expected output: {step['expected_output']}
        Previous step results: {previous_results}
        
        Determine the appropriate parameters to call this tool:
        Return only valid JSON matching the tool's parameter schema.
        """
        
        params_response = await llm.generate(parameter_prompt)
        return json.loads(params_response)
    
    def _resolve_dependencies(self, steps: List[Dict]) -> List[str]:
        """Resolve step dependencies to determine execution order"""
        
        # Simple topological sort
        in_degree = {step["id"]: len(step.get("dependencies", [])) for step in steps}
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for step in steps:
                if current in step.get("dependencies", []):
                    in_degree[step["id"]] -= 1
                    if in_degree[step["id"]] == 0:
                        queue.append(step["id"])
        
        return result
    
    def _add_to_memory(self, memory_type: str, content: Any):
        """Add information to agent memory"""
        memory_entry = {
            "timestamp": time.time(),
            "type": memory_type,
            "content": content
        }
        
        self.working_memory.append(memory_entry)
        
        # Manage memory size (keep last 50 items in working memory)
        if len(self.working_memory) > 50:
            # Move oldest items to long-term memory
            self.long_term_memory.extend(self.working_memory[:10])
            self.working_memory = self.working_memory[10:]
    
    def _get_relevant_memory(self, limit: int = 10) -> str:
        """Get relevant memory for current context"""
        recent_memory = self.working_memory[-limit:]
        
        memory_text = "Recent context:\n"
        for entry in recent_memory:
            memory_text += f"- {entry['type']}: {str(entry['content'])[:200]}...\n"
        
        return memory_text
    
    def _find_tool(self, tool_name: str):
        """Find tool by name"""
        return next((tool for tool in self.tools if tool.name == tool_name), None)
    
    async def _finalize_task(self, execution_results: Dict) -> Dict[str, Any]:
        """Finalize task and create summary"""
        
        finalization_prompt = f"""
        Task: {self.current_task['description']}
        Execution results: {execution_results}
        
        Create a final summary of what was accomplished:
        {{
            "task_completed": true/false,
            "summary": "brief summary of what was achieved",
            "deliverables": ["list of concrete outputs created"],
            "insights": ["key insights or learnings"],
            "recommendations": ["next steps or recommendations"],
            "success_score": 0.0-1.0
        }}
        """
        
        final_summary = await llm.generate(finalization_prompt)
        final_result = json.loads(final_summary)
        
        # Update task record
        self.current_task.update({
            "end_time": time.time(),
            "status": "completed",
            "results": final_result
        })
        
        # Add to long-term memory
        self._add_to_memory("task_completed", {
            "task": self.current_task,
            "results": final_result
        })
        
        return final_result

# Usage example
async def demo_autonomous_agent():
    # Create tools
    research_tool = Tool("web_search", "Search for information", web_search)
    analysis_tool = Tool("analyze_data", "Analyze data", analyze_data)
    
    # Create agent
    agent = AutonomousAgent(
        name="Research Assistant",
        role="research analyst specializing in market trends",
        tools=[research_tool, analysis_tool]
    )
    
    # Give agent a task
    task = "Research the current state of electric vehicle adoption and create a summary report"
    agent.receive_task(task, {"deadline": "today", "audience": "executives"})
    
    # Execute task
    result = await agent.execute_task()
    print(f"Task result: {result}")
```

### 6.2 Reflection Pattern Implementation

Self-improving agents that critique and refine their outputs.

```python
class ReflectiveAgent(AutonomousAgent):
    """Agent with self-reflection and improvement capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_llm = llm  # Main generation model
        self.critic_llm = llm     # Critic model (could be different)
        self.max_reflection_iterations = 3
        
    async def generate_with_reflection(self, task: str) -> Dict[str, Any]:
        """Generate output with iterative self-reflection"""
        
        current_output = None
        reflection_history = []
        
        for iteration in range(self.max_reflection_iterations):
            # Generate or refine output
            if current_output is None:
                # Initial generation
                generation_prompt = f"""
                Task: {task}
                
                Provide a comprehensive response that addresses all aspects of the task.
                Focus on accuracy, completeness, and clarity.
                
                Response:
                """
                current_output = await self.generator_llm.generate(generation_prompt)
            else:
                # Refine based on feedback
                refinement_prompt = f"""
                Task: {task}
                Previous attempt: {current_output}
                Feedback from self-critique: {feedback}
                
                Create an improved version that addresses the feedback:
                
                Improved Response:
                """
                current_output = await self.generator_llm.generate(refinement_prompt)
            
            # Self-critique the output
            feedback = await self._self_critique(task, current_output)
            
            reflection_entry = {
                "iteration": iteration + 1,
                "output": current_output,
                "critique": feedback,
                "satisfactory": self._is_satisfactory(feedback)
            }
            reflection_history.append(reflection_entry)
            
            # Check if satisfactory
            if reflection_entry["satisfactory"]:
                break
        
        return {
            "final_output": current_output,
            "reflection_history": reflection_history,
            "iterations_needed": len(reflection_history),
            "improvement_achieved": len(reflection_history) > 1
        }
    
    async def _self_critique(self, task: str, output: str) -> str:
        """Generate self-critique of the output"""
        
        critique_prompt = f"""
        Original Task: {task}
        Generated Output: {output}
        
        Provide a detailed critique evaluating:
        
        1. ACCURACY & CORRECTNESS
        - Are all facts and statements accurate?
        - Are there any logical errors or inconsistencies?
        - Score: [1-10]
        
        2. COMPLETENESS
        - Does the output fully address all aspects of the task?
        - Are there missing components or topics?
        - Score: [1-10]
        
        3. CLARITY & COHERENCE
        - Is the response clear and well-structured?
        - Is the reasoning easy to follow?
        - Score: [1-10]
        
        4. TASK ADHERENCE
        - Does the output meet the specific requirements?
        - Are format/style requirements followed?
        - Score: [1-10]
        
        OVERALL ASSESSMENT:
        - Overall Score: [Average of above scores]
        - Satisfactory: [Yes/No - Yes if overall score >= 8]
        
        SPECIFIC IMPROVEMENTS NEEDED:
        [List concrete suggestions for improvement]
        
        STRENGTHS TO MAINTAIN:
        [List what works well in the current output]
        """
        
        return await self.critic_llm.generate(critique_prompt)
    
    def _is_satisfactory(self, critique: str) -> bool:
        """Determine if output is satisfactory based on critique"""
        
        # Look for explicit satisfactory indicator
        if "Satisfactory: Yes" in critique:
            return True
        
        # Fallback: analyze sentiment of critique
        positive_indicators = ["good", "excellent", "correct", "complete", "clear", "well-structured"]
        negative_indicators = ["poor", "incorrect", "incomplete", "unclear", "wrong", "missing"]
        
        critique_lower = critique.lower()
        positive_count = sum(indicator in critique_lower for indicator in positive_indicators)
        negative_count = sum(indicator in critique_lower for indicator in negative_indicators)
        
        return positive_count > negative_count

### 6.3 Planning Pattern with Decomposition

Advanced planning agents that break complex tasks into manageable subtasks.

```python
class PlanningAgent(AutonomousAgent):
    """Agent that uses sophisticated planning and decomposition"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner_llm = llm
        self.executor_llm = llm
        
    async def execute_with_planning(self, complex_task: str) -> Dict[str, Any]:
        """Execute complex task using planning decomposition"""
        
        # Phase 1: Analyze task complexity
        task_analysis = await self._analyze_task_complexity(complex_task)
        
        # Phase 2: Generate execution plan
        execution_plan = await self._create_execution_plan(complex_task, task_analysis)
        
        # Phase 3: Execute plan with monitoring
        execution_results = await self._execute_plan_with_monitoring(execution_plan)
        
        # Phase 4: Synthesize final result
        final_result = await self._synthesize_results(complex_task, execution_results)
        
        return {
            "task": complex_task,
            "analysis": task_analysis,
            "plan": execution_plan,
            "execution_results": execution_results,
            "final_result": final_result,
            "success": execution_results.get("overall_success", False)
        }
    
    async def _analyze_task_complexity(self, task: str) -> Dict[str, Any]:
        """Analyze task to understand requirements and complexity"""
        
        analysis_prompt = f"""
        Analyze this complex task in detail:
        
        Task: {task}
        
        Provide comprehensive analysis:
        
        {{
            "complexity_level": "low|medium|high|very_high",
            "estimated_time": "time estimate for completion",
            "required_capabilities": ["list of capabilities needed"],
            "information_requirements": ["what information must be gathered"],
            "potential_challenges": ["anticipated difficulties or obstacles"],
            "success_criteria": ["how to measure successful completion"],
            "decomposition_strategy": "how to break this down effectively",
            "dependencies": ["external dependencies or requirements"],
            "risk_factors": ["potential risks and mitigation needs"]
        }}
        """
        
        analysis_response = await self.planner_llm.generate(analysis_prompt)
        return json.loads(analysis_response)
    
    async def _create_execution_plan(self, task: str, analysis: Dict) -> Dict[str, Any]:
        """Create detailed execution plan based on analysis"""
        
        planning_prompt = f"""
        Task: {task}
        Analysis: {analysis}
        
        Create a detailed execution plan:
        
        {{
            "goal": "clear statement of overall objective",
            "approach": "high-level strategy for task completion",
            "phases": [
                {{
                    "phase_id": "phase_1",
                    "name": "descriptive phase name",
                    "objective": "what this phase accomplishes",
                    "steps": [
                        {{
                            "step_id": "step_1_1",
                            "description": "what to do in this step",
                            "required_tools": ["tools or capabilities needed"],
                            "inputs": ["what inputs this step requires"],
                            "expected_outputs": ["what this step should produce"],
                            "success_criteria": ["how to verify step completion"],
                            "estimated_time": "time estimate",
                            "dependencies": ["other steps that must complete first"]
                        }}
                    ]
                }}
            ],
            "contingency_plans": {{
                "step_failure": "what to do if a step fails",
                "resource_unavailable": "fallback if tools unavailable",
                "time_constraint": "how to adapt if running short on time"
            }},
            "quality_checkpoints": ["points to verify progress and quality"],
            "final_integration": "how to combine all phase outputs"
        }}
        """
        
        plan_response = await self.planner_llm.generate(planning_prompt)
        return json.loads(plan_response)
    
    async def _execute_plan_with_monitoring(self, plan: Dict) -> Dict[str, Any]:
        """Execute plan with progress monitoring and adaptive adjustments"""
        
        execution_state = {
            "overall_success": True,
            "phase_results": {},
            "step_results": {},
            "execution_log": [],
            "adaptations_made": [],
            "total_time": 0
        }
        
        start_time = time.time()
        
        try:
            for phase in plan["phases"]:
                phase_start = time.time()
                phase_result = await self._execute_phase(phase, execution_state)
                phase_time = time.time() - phase_start
                
                execution_state["phase_results"][phase["phase_id"]] = {
                    **phase_result,
                    "execution_time": phase_time
                }
                
                # Check if phase was successful
                if not phase_result.get("success", False):
                    # Attempt contingency plan
                    contingency_result = await self._execute_contingency(phase, execution_state, plan)
                    if not contingency_result.get("success", False):
                        execution_state["overall_success"] = False
                        break
                
                # Log progress
                execution_state["execution_log"].append({
                    "timestamp": time.time(),
                    "phase": phase["phase_id"],
                    "status": "completed" if phase_result.get("success") else "failed",
                    "message": f"Phase {phase['name']} executed"
                })
        
        except Exception as e:
            execution_state["overall_success"] = False
            execution_state["execution_log"].append({
                "timestamp": time.time(),
                "error": str(e),
                "message": "Execution failed with exception"
            })
        
        execution_state["total_time"] = time.time() - start_time
        return execution_state
    
    async def _execute_phase(self, phase: Dict, execution_state: Dict) -> Dict[str, Any]:
        """Execute a single phase of the plan"""
        
        phase_result = {
            "success": True,
            "outputs": {},
            "step_results": {},
            "issues_encountered": []
        }
        
        accumulated_context = ""
        
        for step in phase["steps"]:
            # Check dependencies
            if not await self._check_step_dependencies(step, execution_state):
                phase_result["issues_encountered"].append(f"Dependencies not met for {step['step_id']}")
                continue
            
            # Execute step
            step_result = await self._execute_step_with_context(step, accumulated_context, execution_state)
            
            phase_result["step_results"][step["step_id"]] = step_result
            execution_state["step_results"][step["step_id"]] = step_result
            
            if step_result.get("success", False):
                # Update context for next steps
                accumulated_context += f"\n{step['step_id']}: {step_result.get('output', '')}"
                phase_result["outputs"][step["step_id"]] = step_result.get("output")
            else:
                phase_result["success"] = False
                phase_result["issues_encountered"].append(f"Step {step['step_id']} failed")
        
        return phase_result
    
    async def _execute_step_with_context(self, step: Dict, context: str, execution_state: Dict) -> Dict[str, Any]:
        """Execute individual step with accumulated context"""
        
        step_prompt = f"""
        Step Objective: {step['description']}
        Required Tools: {step.get('required_tools', [])}
        Expected Outputs: {step.get('expected_outputs', [])}
        Success Criteria: {step.get('success_criteria', [])}
        
        Previous Context:
        {context}
        
        Current Execution State:
        - Completed steps: {list(execution_state.get('step_results', {}).keys())}
        - Available outputs: {list(execution_state.get('step_results', {}).keys())}
        
        Execute this step and provide:
        1. The step output/deliverable
        2. Verification that success criteria are met
        3. Any issues or concerns encountered
        
        Step Execution:
        """
        
        step_output = await self.executor_llm.generate(step_prompt)
        
        # Verify step completion
        verification_result = await self._verify_step_completion(step, step_output)
        
        return {
            "success": verification_result["meets_criteria"],
            "output": step_output,
            "verification": verification_result,
            "execution_time": time.time()
        }
    
    async def _verify_step_completion(self, step: Dict, output: str) -> Dict[str, Any]:
        """Verify that step output meets success criteria"""
        
        verification_prompt = f"""
        Step Description: {step['description']}
        Success Criteria: {step.get('success_criteria', [])}
        Expected Outputs: {step.get('expected_outputs', [])}
        Actual Output: {output}
        
        Verify if this step was completed successfully:
        
        {{
            "meets_criteria": true/false,
            "criteria_assessment": {{
                "criterion_1": "assessment of first criterion",
                "criterion_2": "assessment of second criterion"
            }},
            "output_quality": "evaluation of output quality",
            "recommendations": ["suggestions for improvement if needed"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        verification_response = await self.planner_llm.generate(verification_prompt)
        return json.loads(verification_response)

### 6.4 Advanced Tool Use Pattern

Sophisticated tool integration with dynamic tool selection and chaining.

```python
from langchain.tools import DuckDuckGoSearchRun, PythonREPLTool
from langchain.tools.file_management import ReadFileTool, WriteFileTool

class AdvancedToolUseAgent(AutonomousAgent):
    """Agent with sophisticated tool use capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_tools = self._setup_comprehensive_toolset()
        self.tool_usage_history = []
        self.tool_performance_metrics = {}
        
    def _setup_comprehensive_toolset(self) -> Dict[str, Any]:
        """Setup comprehensive set of tools"""
        
        tools = {
            # Information gathering
            "web_search": {
                "tool": DuckDuckGoSearchRun(),
                "description": "Search the web for current information",
                "use_cases": ["current events", "recent information", "factual queries"],
                "cost": "low",
                "reliability": 0.8
            },
            
            # Computation and analysis
            "python_repl": {
                "tool": PythonREPLTool(),
                "description": "Execute Python code for calculations and analysis",
                "use_cases": ["mathematical calculations", "data analysis", "algorithmic problems"],
                "cost": "medium",
                "reliability": 0.95
            },
            
            # File operations
            "read_file": {
                "tool": ReadFileTool(),
                "description": "Read content from files",
                "use_cases": ["document analysis", "data import", "reference lookup"],
                "cost": "low",
                "reliability": 0.9
            },
            
            "write_file": {
                "tool": WriteFileTool(),
                "description": "Write content to files",
                "use_cases": ["report generation", "data export", "result storage"],
                "cost": "low",
                "reliability": 0.9
            },
            
            # Custom tools
            "advanced_calculator": {
                "tool": self._create_advanced_calculator(),
                "description": "Perform complex mathematical calculations",
                "use_cases": ["statistical analysis", "financial calculations", "scientific computing"],
                "cost": "low",
                "reliability": 0.98
            },
            
            "data_analyzer": {
                "tool": self._create_data_analyzer(),
                "description": "Analyze datasets and extract insights",
                "use_cases": ["data exploration", "pattern recognition", "trend analysis"],
                "cost": "medium",
                "reliability": 0.85
            },
            
            "code_generator": {
                "tool": self._create_code_generator(),
                "description": "Generate code in various programming languages",
                "use_cases": ["automation scripts", "utility functions", "algorithm implementation"],
                "cost": "medium",
                "reliability": 0.8
            }
        }
        
        return tools
    
    async def execute_with_intelligent_tool_use(self, task: str) -> Dict[str, Any]:
        """Execute task with intelligent tool selection and usage"""
        
        # Phase 1: Analyze task for tool requirements
        tool_analysis = await self._analyze_tool_requirements(task)
        
        # Phase 2: Create tool usage plan
        tool_plan = await self._create_tool_usage_plan(task, tool_analysis)
        
        # Phase 3: Execute with adaptive tool use
        execution_results = await self._execute_with_adaptive_tools(task, tool_plan)
        
        # Phase 4: Evaluate tool effectiveness
        effectiveness_assessment = await self._assess_tool_effectiveness(execution_results)
        
        return {
            "task": task,
            "tool_analysis": tool_analysis,
            "tool_plan": tool_plan,
            "execution_results": execution_results,
            "effectiveness": effectiveness_assessment,
            "final_output": execution_results.get("final_output")
        }
    
    async def _analyze_tool_requirements(self, task: str) -> Dict[str, Any]:
        """Analyze what tools might be needed for the task"""
        
        analysis_prompt = f"""
        Task: {task}
        
        Available tools and their capabilities:
        {self._format_tool_descriptions()}
        
        Analyze this task to determine:
        
        {{
            "task_category": "information_gathering|computation|analysis|creation|automation",
            "required_capabilities": ["list of needed capabilities"],
            "recommended_tools": ["tools that would be most helpful"],
            "tool_usage_sequence": ["suggested order of tool usage"],
            "potential_challenges": ["challenges that might require tool adaptation"],
            "success_indicators": ["how to measure if tools are effective"],
            "alternative_approaches": ["backup plans if primary tools fail"]
        }}
        """
        
        analysis_response = await self.llm.generate(analysis_prompt)
        return json.loads(analysis_response)
    
    async def _create_tool_usage_plan(self, task: str, analysis: Dict) -> Dict[str, Any]:
        """Create detailed plan for tool usage"""
        
        planning_prompt = f"""
        Task: {task}
        Tool Analysis: {analysis}
        
        Create a detailed tool usage plan:
        
        {{
            "primary_strategy": "main approach for tool usage",
            "tool_sequence": [
                {{
                    "step": 1,
                    "tool": "tool_name",
                    "purpose": "why this tool is needed",
                    "inputs": ["what inputs the tool needs"],
                    "expected_outputs": ["what the tool should produce"],
                    "success_criteria": ["how to know if tool usage was successful"],
                    "fallback_options": ["alternative tools if this fails"]
                }}
            ],
            "integration_strategy": "how to combine outputs from multiple tools",
            "quality_checks": ["checkpoints to verify tool effectiveness"],
            "optimization_opportunities": ["ways to improve tool usage efficiency"]
        }}
        """
        
        plan_response = await self.llm.generate(planning_prompt)
        return json.loads(plan_response)
    
    async def _execute_with_adaptive_tools(self, task: str, plan: Dict) -> Dict[str, Any]:
        """Execute plan with adaptive tool usage"""
        
        execution_context = {
            "task": task,
            "plan": plan,
            "step_outputs": {},
            "tool_performance": {},
            "adaptations_made": [],
            "final_output": None
        }
        
        for step in plan["tool_sequence"]:
            step_num = step["step"]
            tool_name = step["tool"]
            
            # Execute tool with monitoring
            tool_result = await self._execute_tool_with_monitoring(
                tool_name, step, execution_context
            )
            
            execution_context["step_outputs"][step_num] = tool_result
            
            # Evaluate tool performance
            performance = await self._evaluate_tool_performance(tool_result, step)
            execution_context["tool_performance"][tool_name] = performance
            
            # Adapt if necessary
            if performance["effectiveness"] < 0.7:
                adaptation = await self._adapt_tool_usage(step, tool_result, execution_context)
                execution_context["adaptations_made"].append(adaptation)
        
        # Integrate all tool outputs
        execution_context["final_output"] = await self._integrate_tool_outputs(
            task, execution_context["step_outputs"]
        )
        
        return execution_context
    
    async def _execute_tool_with_monitoring(self, tool_name: str, step: Dict, context: Dict) -> Dict[str, Any]:
        """Execute tool with comprehensive monitoring"""
        
        if tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool {tool_name} not available",
                "execution_time": 0
            }
        
        tool_info = self.available_tools[tool_name]
        tool = tool_info["tool"]
        
        # Prepare tool inputs
        tool_inputs = await self._prepare_tool_inputs(step, context)
        
        start_time = time.time()
        
        try:
            # Execute tool
            if hasattr(tool, 'arun'):
                result = await tool.arun(tool_inputs)
            else:
                result = tool.run(tool_inputs)
            
            execution_time = time.time() - start_time
            
            # Record usage
            usage_record = {
                "tool": tool_name,
                "inputs": tool_inputs,
                "result": result,
                "execution_time": execution_time,
                "success": True,
                "timestamp": time.time()
            }
            
            self.tool_usage_history.append(usage_record)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool_used": tool_name,
                "inputs_used": tool_inputs
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            error_record = {
                "tool": tool_name,
                "inputs": tool_inputs,
                "error": str(e),
                "execution_time": execution_time,
                "success": False,
                "timestamp": time.time()
            }
            
            self.tool_usage_history.append(error_record)
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "tool_used": tool_name
            }
    
    def _create_advanced_calculator(self):
        """Create advanced calculator tool"""
        
        def advanced_calculate(expression: str) -> str:
            """Perform advanced mathematical calculations"""
            
            try:
                # Support for various mathematical operations
                import math
                import statistics
                import numpy as np
                
                # Create safe evaluation environment
                safe_dict = {
                    "__builtins__": {},
                    "math": math,
                    "statistics": statistics,
                    "np": np,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "log": math.log,
                    "sqrt": math.sqrt,
                    "pi": math.pi,
                    "e": math.e
                }
                
                result = eval(expression, safe_dict)
                return f"Result: {result}"
                
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        return advanced_calculate
    
    def _create_data_analyzer(self):
        """Create data analysis tool"""
        
        def analyze_data(data_description: str) -> str:
            """Analyze data and provide insights"""
            
            analysis_prompt = f"""
            Data to analyze: {data_description}
            
            Provide analysis including:
            1. Data characteristics and patterns
            2. Key insights and findings
            3. Potential correlations or trends
            4. Recommendations based on the data
            5. Statistical summary if applicable
            
            Analysis:
            """
            
            # In practice, this would integrate with actual data analysis libraries
            return f"Data analysis for: {data_description}"
        
        return analyze_data
    
    def _create_code_generator(self):
        """Create code generation tool"""
        
        def generate_code(requirements: str) -> str:
            """Generate code based on requirements"""
            
            code_prompt = f"""
            Code requirements: {requirements}
            
            Generate clean, well-documented code that:
            1. Meets the specified requirements
            2. Includes error handling
            3. Has clear variable names and structure
            4. Includes usage examples if applicable
            
            Generated code:
            """
            
            # In practice, this would use a code generation model
            return f"Generated code for: {requirements}"
        
        return generate_code
```

---

## Level 7: Multi-Agent Coordination

### What You'll Learn
- Agent communication protocols
- Collaborative problem solving
- Workflow orchestration across agents
- Conflict resolution and consensus building

### What You Can Build After This Level
✅ Complex collaborative AI systems  
✅ Distributed problem-solving networks  
✅ Specialized agent teams  
✅ Self-organizing AI workflows  

### 7.1 Agent Communication Infrastructure

```python
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COLLABORATION_REQUEST = "collaboration_request"
    STATUS_UPDATE = "status_update"
    HELP_REQUEST = "help_request"
    NEGOTIATION = "negotiation"
    CONSENSUS_BUILDING = "consensus_building"

@dataclass
class Message:
    id: str
    sender_id: str
    receiver_id: str  # or "broadcast" for all agents
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1=low, 5=high
    requires_response: bool = False
    conversation_id: Optional[str] = None

class MessageBroker:
    def __init__(self):
        self.agents: Dict[str, 'CollaborativeAgent'] = {}
        self.message_queue: List[Message] = []
        self.message_history: List[Message] = []
        self.active_conversations: Dict[str, List[Message]] = {}
        self.running = False
        
    def register_agent(self, agent: 'CollaborativeAgent'):
        """Register an agent with the broker"""
        self.agents[agent.agent_id] = agent
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    async def send_message(self, message: Message):
        """Send a message to target agent(s)"""
        message.timestamp = time.time()
        
        self.message_history.append(message)
        
        # Add to conversation history
        if message.conversation_id:
            if message.conversation_id not in self.active_conversations:
                self.active_conversations[message.conversation_id] = []
            self.active_conversations[message.conversation_id].append(message)
        
        # Deliver message
        if message.receiver_id == "broadcast":
            # Broadcast to all agents except sender
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender_id:
                    await agent.receive_message(message)
        else:
            # Send to specific agent
            if message.receiver_id in self.agents:
                await self.agents[message.receiver_id].receive_message(message)
            else:
                print(f"Warning: Agent {message.receiver_id} not found")
    
    async def start_message_processing(self):
        """Start processing message queue"""
        self.running = True
        while self.running:
            # Process any queued messages
            while self.message_queue:
                message = self.message_queue.pop(0)
                await self.send_message(message)
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    def stop_message_processing(self):
        """Stop message processing"""
        self.running = False
    
    def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """Get history for a specific conversation"""
        return self.active_conversations.get(conversation_id, [])
    
    def get_agent_message_history(self, agent_id: str, limit: int = 50) -> List[Message]:
        """Get message history for a specific agent"""
        agent_messages = [
            msg for msg in self.message_history[-limit:]
            if msg.sender_id == agent_id or msg.receiver_id == agent_id
        ]
        return agent_messages

class CollaborativeAgent(AdvancedAgent):
    def __init__(self, name: str, role: str, broker: MessageBroker, specializations: List[str] = None, **kwargs):
        super().__init__(name, role, **kwargs)
        self.broker = broker
        self.specializations = specializations or []
        self.collaborations: Dict[str, Dict] = {}  # Active collaborations
        self.trust_scores: Dict[str, float] = {}  # Trust scores for other agents
        self.workload = 0  # Current workload (0-100)
        
        # Register with broker
        self.broker.register_agent(self)
        
        # Message handling
        self.message_handlers = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.INFORMATION_SHARE: self._handle_information_share,
            MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            MessageType.HELP_REQUEST: self._handle_help_request,
            MessageType.NEGOTIATION: self._handle_negotiation,
            MessageType.CONSENSUS_BUILDING: self._handle_consensus_building
        }
    
    async def receive_message(self, message: Message):
        """Receive and process a message"""
        
        # Add to working memory
        self._add_to_memory("message_received", {
            "from": message.sender_id,
            "type": message.message_type.value,
            "content": message.content
        })
        
        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            await handler(message)
        else:
            print(f"No handler for message type: {message.message_type}")
    
    async def _handle_task_request(self, message: Message):
        """Handle incoming task requests"""
        
        task_content = message.content
        
        # Evaluate if I can and should take this task
        evaluation = await self._evaluate_task_request(task_content)
        
        if evaluation["can_accept"]:
            # Accept the task
            response = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    "response": "accepted",
                    "estimated_completion": evaluation["estimated_time"],
                    "confidence": evaluation["confidence"],
                    "conditions": evaluation.get("conditions", [])
                },
                conversation_id=message.conversation_id
            )
            
            # Start working on the task
            self.workload += evaluation["workload_impact"]
            await self._start_collaborative_task(task_content, message.sender_id)
            
        else:
            # Decline the task
            response = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    "response": "declined",
                    "reason": evaluation["decline_reason"],
                    "alternative_suggestions": evaluation.get("alternatives", [])
                },
                conversation_id=message.conversation_id
            )
        
        await self.broker.send_message(response)
    
    async def _evaluate_task_request(self, task_content: Dict) -> Dict[str, Any]:
        """Evaluate whether to accept a task request"""
        
        evaluation_prompt = f"""
        Task request: {task_content}
        My role: {self.role}
        My specializations: {self.specializations}
        Current workload: {self.workload}/100
        Available tools: {[tool.name for tool in self.tools]}
        
        Evaluate this task request:
        {{
            "can_accept": true/false,
            "confidence": 0.0-1.0,
            "estimated_time": "time estimate",
            "workload_impact": 1-50,
            "decline_reason": "reason if declining",
            "conditions": ["any conditions for acceptance"],
            "alternatives": ["suggest other agents or approaches if declining"]
        }}
        """
        
        evaluation = await llm.generate(evaluation_prompt)
        return json.loads(evaluation)
    
    async def _handle_information_share(self, message: Message):
        """Handle information sharing from other agents"""
        
        info_content = message.content
        
        # Process and store the shared information
        self._add_to_memory("shared_information", {
            "from_agent": message.sender_id,
            "information": info_content,
            "relevance": await self._assess_information_relevance(info_content)
        })
        
        # Send acknowledgment if requested
        if message.requires_response:
            ack_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.TASK_RESPONSE,
                content={"status": "information_received", "relevance_score": await self._assess_information_relevance(info_content)},
                conversation_id=message.conversation_id
            )
            await self.broker.send_message(ack_message)
    
    async def _handle_collaboration_request(self, message: Message):
        """Handle collaboration requests"""
        
        collab_request = message.content
        
        # Evaluate collaboration opportunity
        evaluation = await self._evaluate_collaboration(collab_request, message.sender_id)
        
        if evaluation["accept"]:
            # Start collaboration
            collaboration_id = str(uuid.uuid4())
            self.collaborations[collaboration_id] = {
                "partner_agent": message.sender_id,
                "type": collab_request["collaboration_type"],
                "objective": collab_request["objective"],
                "my_role": evaluation["my_role"],
                "start_time": time.time(),
                "status": "active"
            }
            
            response_content = {
                "response": "accepted",
                "collaboration_id": collaboration_id,
                "my_role": evaluation["my_role"],
                "proposed_approach": evaluation["approach"]
            }
        else:
            response_content = {
                "response": "declined",
                "reason": evaluation["reason"]
            }
        
        response = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            content=response_content,
            conversation_id=message.conversation_id
        )
        
        await self.broker.send_message(response)
    
    async def initiate_collaboration(self, target_agent_id: str, collaboration_type: str, objective: str) -> str:
        """Initiate collaboration with another agent"""
        
        conversation_id = str(uuid.uuid4())
        
        collab_message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=MessageType.COLLABORATION_REQUEST,
            content={
                "collaboration_type": collaboration_type,
                "objective": objective,
                "my_capabilities": self.specializations,
                "expected_duration": "TBD",
                "urgency": "medium"
            },
            conversation_id=conversation_id,
            requires_response=True
        )
        
        await self.broker.send_message(collab_message)
        return conversation_id
    
    async def broadcast_information(self, information: Dict[str, Any], information_type: str):
        """Broadcast information to all other agents"""
        
        broadcast_message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id="broadcast",
            message_type=MessageType.INFORMATION_SHARE,
            content={
                "information_type": information_type,
                "data": information,
                "source": self.agent_id,
                "confidence": 0.8  # Default confidence
            }
        )
        
        await self.broker.send_message(broadcast_message)
    
    async def request_help(self, help_type: str, context: Dict[str, Any], preferred_agents: List[str] = None):
        """Request help from other agents"""
        
        if preferred_agents:
            # Send to specific agents
            for agent_id in preferred_agents:
                help_message = Message(
                    id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=agent_id,
                    message_type=MessageType.HELP_REQUEST,
                    content={
                        "help_type": help_type,
                        "context": context,
                        "urgency": context.get("urgency", "medium")
                    },
                    requires_response=True
                )
                await self.broker.send_message(help_message)
        else:
            # Broadcast help request
            help_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id="broadcast",
                message_type=MessageType.HELP_REQUEST,
                content={
                    "help_type": help_type,
                    "context": context,
                    "urgency": context.get("urgency", "medium")
                },
                requires_response=True
            )
            await self.broker.send_message(help_message)

### 7.2 Specialized Agent Types

class CoordinatorAgent(CollaborativeAgent):
    """Agent specialized in coordinating multi-agent workflows"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_workflows: Dict[str, Dict] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_workloads: Dict[str, int] = {}
        
    async def orchestrate_complex_task(self, task_description: str, requirements: Dict = None) -> str:
        """Orchestrate a complex task across multiple agents"""
        
        workflow_id = str(uuid.uuid4())
        
        # Analyze task and create workflow
        workflow = await self._create_workflow(task_description, requirements)
        
        # Find and assign agents
        assignments = await self._assign_agents_to_tasks(workflow["subtasks"])
        
        # Start workflow execution
        self.active_workflows[workflow_id] = {
            "description": task_description,
            "workflow": workflow,
            "assignments": assignments,
            "status": "in_progress",
            "start_time": time.time(),
            "completed_subtasks": {},
            "failed_subtasks": {}
        }
        
        # Send initial task assignments
        await self._send_initial_assignments(workflow_id, assignments)
        
        return workflow_id
    
    async def _create_workflow(self, task_description: str, requirements: Dict) -> Dict[str, Any]:
        """Create workflow breakdown for complex task"""
        
        workflow_prompt = f"""
        Complex task: {task_description}
        Requirements: {requirements or {}}
        Available agent types: research, analysis, writing, technical, creative
        
        Break down into a coordinated workflow:
        {{
            "objective": "overall goal",
            "subtasks": [
                {{
                    "id": "subtask_1",
                    "description": "what needs to be done",
                    "required_capabilities": ["list of needed skills"],
                    "dependencies": ["list of subtask ids that must complete first"],
                    "estimated_effort": "low|medium|high",
                    "deliverable": "what this subtask should produce",
                    "success_criteria": "how to measure success"
                }}
            ],
            "coordination_points": ["when agents need to sync"],
            "final_integration": "how to combine all results"
        }}
        """
        
        workflow_response = await llm.generate(workflow_prompt)
        return json.loads(workflow_response)
    
    async def _assign_agents_to_tasks(self, subtasks: List[Dict]) -> Dict[str, str]:
        """Assign optimal agents to each subtask"""
        
        assignments = {}
        
        for subtask in subtasks:
            # Find best agent for this subtask
            best_agent = await self._find_optimal_agent(subtask["required_capabilities"])
            
            if best_agent:
                assignments[subtask["id"]] = best_agent
                # Update agent workload tracking
                self.agent_workloads[best_agent] = self.agent_workloads.get(best_agent, 0) + 20
            else:
                assignments[subtask["id"]] = None  # No suitable agent found
        
        return assignments
    
    async def _find_optimal_agent(self, required_capabilities: List[str]) -> Optional[str]:
        """Find the best agent for required capabilities"""
        
        best_agent = None
        best_score = 0
        
        for agent_id, agent in self.broker.agents.items():
            if agent_id == self.agent_id:  # Don't assign to self
                continue
            
            # Calculate capability match score
            agent_capabilities = getattr(agent, 'specializations', [])
            capability_match = len(set(required_capabilities) & set(agent_capabilities))
            
            # Factor in current workload (prefer less busy agents)
            current_workload = self.agent_workloads.get(agent_id, 0)
            workload_penalty = current_workload / 100
            
            # Calculate overall score
            score = capability_match - workload_penalty
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    async def monitor_workflow_progress(self, workflow_id: str) -> Dict[str, Any]:
        """Monitor and manage workflow progress"""
        
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        # Check completion status
        total_subtasks = len(workflow["workflow"]["subtasks"])
        completed_count = len(workflow["completed_subtasks"])
        failed_count = len(workflow["failed_subtasks"])
        
        progress = {
            "workflow_id": workflow_id,
            "total_subtasks": total_subtasks,
            "completed": completed_count,
            "failed": failed_count,
            "in_progress": total_subtasks - completed_count - failed_count,
            "completion_percentage": (completed_count / total_subtasks) * 100,
            "status": workflow["status"]
        }
        
        # Check if workflow is complete
        if completed_count == total_subtasks:
            workflow["status"] = "completed"
            await self._finalize_workflow(workflow_id)
        elif failed_count > total_subtasks / 2:  # Too many failures
            workflow["status"] = "failed"
            await self._handle_workflow_failure(workflow_id)
        
        return progress

class ResearchAgent(CollaborativeAgent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specializations = ["research", "information_gathering", "fact_checking", "data_collection"]
        self.research_sources = ["web", "databases", "documents", "apis"]
        
    async def conduct_comprehensive_research(self, topic: str, depth: str = "thorough") -> Dict[str, Any]:
        """Conduct comprehensive research on a topic"""
        
        research_plan = await self._create_research_plan(topic, depth)
        
        findings = {}
        for phase in research_plan["phases"]:
            phase_results = await self._execute_research_phase(phase)
            findings[phase["name"]] = phase_results
            
            # Share interim findings with interested agents
            await self.broadcast_information(
                {"topic": topic, "phase": phase["name"], "findings": phase_results},
                "research_update"
            )
        
        # Synthesize all findings
        synthesis = await self._synthesize_research_findings(topic, findings)
        
        final_report = {
            "topic": topic,
            "research_plan": research_plan,
            "detailed_findings": findings,
            "synthesis": synthesis,
            "confidence_score": synthesis.get("confidence", 0.8),
            "sources_consulted": self._extract_sources(findings)
        }
        
        return final_report
    
    async def _create_research_plan(self, topic: str, depth: str) -> Dict[str, Any]:
        """Create structured research plan"""
        
        planning_prompt = f"""
        Research topic: {topic}
        Research depth: {depth}
        Available sources: {self.research_sources}
        
        Create a comprehensive research plan:
        {{
            "research_question": "main question to answer",
            "sub_questions": ["supporting questions"],
            "phases": [
                {{
                    "name": "phase_name",
                    "objective": "what to accomplish",
                    "methods": ["research methods to use"],
                    "sources": ["types of sources to consult"],
                    "deliverable": "what this phase produces"
                }}
            ],
            "success_criteria": "how to measure research quality"
        }}
        """
        
        plan_response = await llm.generate(planning_prompt)
        return json.loads(plan_response)

class AnalysisAgent(CollaborativeAgent):
    """Agent specialized in data analysis and insights"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.specializations = ["data_analysis", "statistical_analysis", "pattern_recognition", "insight_generation"]
        self.analysis_types = ["descriptive", "diagnostic", "predictive", "prescriptive"]
        
    async def analyze_research_data(self, research_data: Dict, analysis_objectives: List[str]) -> Dict[str, Any]:
        """Analyze research data and generate insights"""
        
        # Determine appropriate analysis approach
        analysis_approach = await self._determine_analysis_approach(research_data, analysis_objectives)
        
        # Perform analysis
        analysis_results = {}
        for analysis_type in analysis_approach["recommended_analyses"]:
            result = await self._perform_analysis(research_data, analysis_type, analysis_objectives)
            analysis_results[analysis_type] = result
        
        # Generate insights
        insights = await self._generate_insights(research_data, analysis_results, analysis_objectives)
        
        return {
            "analysis_approach": analysis_approach,
            "analysis_results": analysis_results,
            "insights": insights,
            "recommendations": insights.get("recommendations", []),
            "confidence_level": insights.get("confidence", 0.7)
        }

### 7.3 Consensus Building and Conflict Resolution

class ConsensusBuildingMixin:
    """Mixin for agents that participate in consensus building"""
    
    async def initiate_consensus_process(self, topic: str, options: List[Dict], participants: List[str]) -> str:
        """Initiate a consensus building process"""
        
        consensus_id = str(uuid.uuid4())
        
        # Send consensus building request to all participants
        for participant in participants:
            consensus_message = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=participant,
                message_type=MessageType.CONSENSUS_BUILDING,
                content={
                    "consensus_id": consensus_id,
                    "topic": topic,
                    "options": options,
                    "initiator": self.agent_id,
                    "deadline": time.time() + 3600,  # 1 hour deadline
                    "phase": "initial_positions"
                },
                conversation_id=consensus_id,
                requires_response=True
            )
            await self.broker.send_message(consensus_message)
        
        return consensus_id
    
    async def _handle_consensus_building(self, message: Message):
        """Handle consensus building messages"""
        
        consensus_content = message.content
        phase = consensus_content.get("phase", "initial_positions")
        
        if phase == "initial_positions":
            # Provide initial position on the topic
            position = await self._determine_position(consensus_content["topic"], consensus_content["options"])
            
            response = Message(
                id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.CONSENSUS_BUILDING,
                content={
                    "consensus_id": consensus_content["consensus_id"],
                    "phase": "position_response",
                    "position": position,
                    "reasoning": position["reasoning"],
                    "openness_to_change": position["flexibility"]
                },
                conversation_id=message.conversation_id
            )
            
            await self.broker.send_message(response)
            
        elif phase == "negotiation":
            # Participate in negotiation phase
            await self._participate_in_negotiation(message)
            
        elif phase == "final_decision":
            # Accept or reject final consensus
            await self._respond_to_final_consensus(message)
    
    async def _determine_position(self, topic: str, options: List[Dict]) -> Dict[str, Any]:
        """Determine agent's position on a topic"""
        
        position_prompt = f"""
        Topic: {topic}
        Options: {options}
        My role: {self.role}
        My specializations: {self.specializations}
        
        Determine my position:
        {{
            "preferred_option": "option_id",
            "reasoning": "why I prefer this option",
            "concerns": ["issues with other options"],
            "flexibility": 0.0-1.0,
            "non_negotiables": ["aspects I won't compromise on"],
            "alternative_suggestions": ["modifications or new options"]
        }}
        """
        
        position_response = await llm.generate(position_prompt)
        return json.loads(position_response)

### 7.4 Multi-Agent System Integration

class MultiAgentSystem:
    """Complete multi-agent system with coordination and management"""
    
    def __init__(self):
        self.broker = MessageBroker()
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.system_coordinator = None
        self.active_workflows: Dict[str, Dict] = {}
        
    def add_agent(self, agent: CollaborativeAgent):
        """Add an agent to the system"""
        self.agents[agent.agent_id] = agent
        
    def set_coordinator(self, coordinator: CoordinatorAgent):
        """Set the system coordinator"""
        self.system_coordinator = coordinator
        
    async def execute_complex_project(self, project_description: str, requirements: Dict = None) -> Dict[str, Any]:
        """Execute a complex project using multiple agents"""
        
        if not self.system_coordinator:
            return {"error": "No coordinator agent available"}
        
        # Coordinator orchestrates the project
        workflow_id = await self.system_coordinator.orchestrate_complex_task(
            project_description, requirements
        )
        
        # Monitor progress
        return await self._monitor_project_execution(workflow_id)
    
    async def _monitor_project_execution(self, workflow_id: str) -> Dict[str, Any]:
        """Monitor project execution until completion"""
        
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            progress = await self.system_coordinator.monitor_workflow_progress(workflow_id)
            
            if progress["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)  # Check every second
            iteration += 1
        
        return progress
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        
        agent_metrics = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_performance_metrics'):
                agent_metrics[agent_id] = agent.get_performance_metrics()
        
        total_messages = len(self.broker.message_history)
        active_conversations = len(self.broker.active_conversations)
        
        return {
            "total_agents": len(self.agents),
            "agent_metrics": agent_metrics,
            "communication_metrics": {
                "total_messages": total_messages,
                "active_conversations": active_conversations,
                "message_types": self._get_message_type_distribution()
            },
            "system_uptime": time.time() - getattr(self, 'start_time', time.time())
        }
    
    def _get_message_type_distribution(self) -> Dict[str, int]:
        """Get distribution of message types"""
        distribution = {}
        for message in self.broker.message_history:
            msg_type = message.message_type.value
            distribution[msg_type] = distribution.get(msg_type, 0) + 1
        return distribution

# Usage example
async def demo_multi_agent_system():
    # Create multi-agent system
    system = MultiAgentSystem()
    
    # Create specialized agents
    coordinator = CoordinatorAgent(
        name="Project Coordinator",
        role="workflow orchestrator",
        broker=system.broker,
        specializations=["coordination", "project_management"]
    )
    
    researcher = ResearchAgent(
        name="Research Specialist", 
        role="information gatherer",
        broker=system.broker
    )
    
    analyst = AnalysisAgent(
        name="Data Analyst",
        role="insight generator", 
        broker=system.broker
    )
    
    # Add agents to system
    system.add_agent(coordinator)
    system.add_agent(researcher)
    system.add_agent(analyst)
    system.set_coordinator(coordinator)
    
    # Start message processing
    asyncio.create_task(system.broker.start_message_processing())
    
    # Execute complex project
    project = "Analyze the impact of AI adoption on employment in the technology sector"
    requirements = {
        "depth": "comprehensive",
        "timeframe": "2020-2024", 
        "deliverable": "executive report"
    }
    
    result = await system.execute_complex_project(project, requirements)
    print(f"Project result: {result}")
    
    # Get system metrics
    metrics = system.get_system_metrics()
    print(f"System metrics: {metrics}")

---

## Level 8: Production-Grade Systems

### What You'll Learn
- Scalability and performance optimization
- Error handling and resilience patterns
- Monitoring and observability
- Security and safety considerations

### What You Can Build After This Level
✅ Enterprise-grade AI applications  
✅ Fault-tolerant distributed systems  
✅ Scalable multi-tenant platforms  
✅ Production-ready AI services  

### 8.1 Scalability and Performance

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import redis
from dataclasses import dataclass
from typing import Optional
import hashlib

@dataclass
class CacheConfig:
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum cache size
    enabled: bool = True

class ProductionAgent(CollaborativeAgent):
    """Production-ready agent with caching, monitoring, and error handling"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Performance optimizations
        self.cache = redis.Redis(host='localhost', port=6379, db=0) if self._redis_available() else {}
        self.cache_config = CacheConfig()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Monitoring
        self.metrics = {
            "requests_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_encountered": 0,
            "average_response_time": 0,
            "total_response_time": 0
        }
        
        # Circuit breaker for external services
        self.circuit_breakers = {}
        
    def _redis_available(self) -> bool:
        """Check if Redis is available"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            return True
        except:
            return False
    
    async def process_request_with_caching(self, request: str, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Process request with caching support"""
        
        start_time = time.time()
        
        try:
            # Generate cache key if not provided
            if not cache_key:
                cache_key = self._generate_cache_key(request)
            
            # Try to get from cache
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Process request
            result = await self._process_request_internal(request)
            
            # Cache the result
            await self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.metrics["errors_encountered"] += 1
            raise e
            
        finally:
            # Update metrics
            response_time = time.time() - start_time
            self.metrics["requests_processed"] += 1
            self.metrics["total_response_time"] += response_time
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["requests_processed"]
            )
    
    def _generate_cache_key(self, request: str) -> str:
        """Generate cache key from request"""
        # Include agent version and capabilities in key
        key_data = f"{self.agent_id}:{request}:{hash(str(self.specializations))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get result from cache"""
        if not self.cache_config.enabled:
            return None
            
        try:
            if isinstance(self.cache, dict):
                # In-memory cache
                return self.cache.get(cache_key)
            else:
                # Redis cache
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return json.loads(cached_data.decode())
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            
        return None
    
    async def _set_cache(self, cache_key: str, result: Dict):
        """Set result in cache"""
        if not self.cache_config.enabled:
            return
            
        try:
            if isinstance(self.cache, dict):
                # In-memory cache with size limit
                if len(self.cache) >= self.cache_config.max_size:
                    # Remove oldest entry (simplified LRU)
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result
            else:
                # Redis cache
                self.cache.setex(
                    cache_key, 
                    self.cache_config.ttl, 
                    json.dumps(result)
                )
        except Exception as e:
            print(f"Cache storage error: {e}")

class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

### 8.2 Comprehensive Error Handling

class ResilientAgent(ProductionAgent):
    """Agent with comprehensive error handling and recovery"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_config = {
            "max_retries": 3,
            "backoff_factor": 2,
            "retry_exceptions": [ConnectionError, TimeoutError]
        }
        self.fallback_strategies = {}
        
    async def execute_with_resilience(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute operation with full resilience patterns"""
        
        operation_id = str(uuid.uuid4())
        
        try:
            # Try primary execution path
            result = await self._execute_with_retry(operation, *args, **kwargs)
            
            return {
                "success": True,
                "result": result,
                "operation_id": operation_id,
                "execution_path": "primary"
            }
            
        except Exception as primary_error:
            
            # Try fallback strategy
            try:
                fallback_result = await self._execute_fallback(operation, primary_error, *args, **kwargs)
                
                return {
                    "success": True,
                    "result": fallback_result,
                    "operation_id": operation_id,
                    "execution_path": "fallback",
                    "primary_error": str(primary_error)
                }
                
            except Exception as fallback_error:
                
                # Final error handling
                error_report = await self._handle_final_error(
                    operation, primary_error, fallback_error, operation_id
                )
                
                return {
                    "success": False,
                    "operation_id": operation_id,
                    "errors": {
                        "primary": str(primary_error),
                        "fallback": str(fallback_error)
                    },
                    "error_report": error_report
                }
    
    async def _execute_with_retry(self, operation: str, *args, **kwargs):
        """Execute operation with retry logic"""
        
        max_retries = self.retry_config["max_retries"]
        backoff_factor = self.retry_config["backoff_factor"]
        
        for attempt in range(max_retries + 1):
            try:
                if operation == "llm_generation":
                    return await self._safe_llm_generation(*args, **kwargs)
                elif operation == "tool_execution":
                    return await self._safe_tool_execution(*args, **kwargs)
                elif operation == "collaboration":
                    return await self._safe_collaboration(*args, **kwargs)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                    
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                # Check if this exception type should be retried
                if not any(isinstance(e, exc_type) for exc_type in self.retry_config["retry_exceptions"]):
                    raise e
                
                # Calculate backoff delay
                delay = backoff_factor ** attempt
                await asyncio.sleep(delay)
                
                print(f"Retry {attempt + 1}/{max_retries} for {operation} after {delay}s delay")
    
    async def _execute_fallback(self, operation: str, primary_error: Exception, *args, **kwargs):
        """Execute fallback strategy when primary fails"""
        
        if operation == "llm_generation":
            # Fallback: Use simpler prompt or cached response
            return await self._fallback_llm_generation(primary_error, *args, **kwargs)
            
        elif operation == "tool_execution":
            # Fallback: Use alternative tool or manual process
            return await self._fallback_tool_execution(primary_error, *args, **kwargs)
            
        elif operation == "collaboration":
            # Fallback: Work independently or request human help
            return await self._fallback_collaboration(primary_error, *args, **kwargs)
        
        else:
            raise Exception(f"No fallback strategy for operation: {operation}")
    
    async def _fallback_llm_generation(self, error: Exception, prompt: str, **kwargs):
        """Fallback strategy for LLM generation failures"""
        
        # Try with simplified prompt
        simplified_prompt = f"Briefly: {prompt[:500]}"  # Truncate and simplify
        
        try:
            return await llm.generate(simplified_prompt)
        except:
            # Final fallback: return templated response
            return f"Unable to process request due to system limitations. Error: {str(error)[:100]}"
    
    async def _safe_llm_generation(self, prompt: str, **kwargs):
        """Safely generate LLM response"""
        
        # Validate prompt length
        if len(prompt) > 10000:  # Arbitrary limit
            raise ValueError("Prompt too long")
        
        # Add safety instructions
        safe_prompt = f"""
        {prompt}
        
        Important: Provide a helpful, accurate, and safe response. If you cannot answer safely or accurately, explain why.
        """
        
        return await llm.generate(safe_prompt)

### 8.3 Monitoring and Observability

class ObservableAgent(ResilientAgent):
    """Agent with comprehensive monitoring and observability"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = TelemetryCollector(self.agent_id)
        self.health_status = "healthy"
        self.last_health_check = time.time()
        
    async def execute_with_observability(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute operation with full observability"""
        
        # Start telemetry collection
        span_id = self.telemetry.start_span(operation)
        
        try:
            result = await self.execute_with_resilience(operation, *args, **kwargs)
            
            # Record success metrics
            self.telemetry.record_success(span_id, result)
            
            return result
            
        except Exception as e:
            # Record error metrics
            self.telemetry.record_error(span_id, e)
            raise e
            
        finally:
            # End telemetry collection
            self.telemetry.end_span(span_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        
        health_report = {
            "agent_id": self.agent_id,
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check memory usage
        memory_check = self._check_memory_health()
        health_report["checks"]["memory"] = memory_check
        
        # Check tool availability
        tools_check = await self._check_tools_health()
        health_report["checks"]["tools"] = tools_check
        
        # Check external dependencies
        deps_check = await self._check_dependencies_health()
        health_report["checks"]["dependencies"] = deps_check
        
        # Check performance metrics
        perf_check = self._check_performance_health()
        health_report["checks"]["performance"] = perf_check
        
        # Determine overall health
        all_checks = [memory_check, tools_check, deps_check, perf_check]
        if any(check["status"] == "unhealthy" for check in all_checks):
            health_report["status"] = "unhealthy"
        elif any(check["status"] == "degraded" for check in all_checks):
            health_report["status"] = "degraded"
        
        self.health_status = health_report["status"]
        self.last_health_check = time.time()
        
        return health_report
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage health"""
        
        working_memory_size = len(self.working_memory)
        long_term_memory_size = len(self.long_term_memory)
        
        if working_memory_size > 100:
            status = "degraded"
            message = "Working memory size is high"
        elif working_memory_size > 200:
            status = "unhealthy"
            message = "Working memory size is critical"
        else:
            status = "healthy"
            message = "Memory usage is normal"
        
        return {
            "status": status,
            "message": message,
            "metrics": {
                "working_memory_size": working_memory_size,
                "long_term_memory_size": long_term_memory_size
            }
        }
    
    async def _check_tools_health(self) -> Dict[str, Any]:
        """Check tool availability and health"""
        
        tool_statuses = {}
        overall_status = "healthy"
        
        for tool in self.tools:
            try:
                # Test tool with a simple operation
                test_result = await tool.execute({"test": True})
                tool_statuses[tool.name] = "healthy"
            except Exception as e:
                tool_statuses[tool.name] = f"unhealthy: {str(e)}"
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "tool_statuses": tool_statuses
        }
    
    async def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check external dependencies health"""
        
        dependencies = {
            "llm_service": await self._check_llm_health(),
            "cache_service": self._check_cache_health(),
            "message_broker": self._check_broker_health()
        }
        
        if any(status != "healthy" for status in dependencies.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "dependencies": dependencies
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance and health metrics"""
        
        return {
            "agent_info": {
                "id": self.agent_id,
                "name": self.name,
                "role": self.role,
                "specializations": self.specializations
            },
            "health": {
                "status": self.health_status,
                "last_check": self.last_health_check
            },
            "performance": self.metrics,
            "telemetry": self.telemetry.get_summary(),
            "cache_stats": {
                "hits": self.metrics["cache_hits"],
                "misses": self.metrics["cache_misses"],
                "hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            }
        }

class TelemetryCollector:
    """Collect and manage telemetry data"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.active_spans = {}
        self.completed_spans = []
        
    def start_span(self, operation: str) -> str:
        """Start a new telemetry span"""
        span_id = str(uuid.uuid4())
        
        self.active_spans[span_id] = {
            "operation": operation,
            "start_time": time.time(),
            "agent_id": self.agent_id
        }
        
        return span_id
    
    def end_span(self, span_id: str):
        """End a telemetry span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span["end_time"] = time.time()
            span["duration"] = span["end_time"] - span["start_time"]
            
            self.completed_spans.append(span)
            del self.active_spans[span_id]
    
    def record_success(self, span_id: str, result: Dict):
        """Record successful operation"""
        if span_id in self.active_spans:
            self.active_spans[span_id]["success"] = True
            self.active_spans[span_id]["result_size"] = len(str(result))
    
    def record_error(self, span_id: str, error: Exception):
        """Record operation error"""
        if span_id in self.active_spans:
            self.active_spans[span_id]["success"] = False
            self.active_spans[span_id]["error"] = str(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary"""
        if not self.completed_spans:
            return {"message": "No completed operations"}
        
        total_operations = len(self.completed_spans)
        successful_operations = sum(1 for span in self.completed_spans if span.get("success", False))
        
        durations = [span["duration"] for span in self.completed_spans if "duration" in span]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_operations": total_operations,
            "success_rate": successful_operations / total_operations,
            "average_duration": avg_duration,
            "operations_by_type": self._get_operation_distribution()
        }
    
    def _get_operation_distribution(self) -> Dict[str, int]:
        """Get distribution of operation types"""
        distribution = {}
        for span in self.completed_spans:
            operation = span["operation"]
            distribution[operation] = distribution.get(operation, 0) + 1
        return distribution

---

## Working with Small LLM Models

### Understanding Small Model Limitations and Advantages

Small LLM models (1B-7B parameters) have unique characteristics that require adapted approaches:

**Advantages:**
- ✅ Faster inference times
- ✅ Lower memory requirements  
- ✅ Better cost efficiency
- ✅ Easier to deploy locally
- ✅ Better privacy control

**Limitations:**
- ⚠️ Reduced reasoning capabilities
- ⚠️ Smaller context windows
- ⚠️ Less general knowledge
- ⚠️ More sensitive to prompt formatting
- ⚠️ Limited multilingual support

### 10.1 Optimized Prompt Engineering for Small Models

```python
class SmallModelOptimizer:
    """Optimize prompts and workflows for small LLM models"""
    
    def __init__(self, model_name: str, context_limit: int = 2048):
        self.model_name = model_name
        self.context_limit = context_limit
        self.prompt_templates = self._load_optimized_templates()
        
    def _load_optimized_templates(self) -> Dict[str, str]:
        """Load prompt templates optimized for small models"""
        return {
            "classification": """Task: Classify this text
Text: {text}
Categories: {categories}
Answer: """,
            
            "extraction": """Extract {target} from:
{text}

{target}: """,
            
            "summary": """Summarize in {length} words:
{text}

Summary: """,
            
            "qa": """Context: {context}
Question: {question}
Answer: """,
            
            "reasoning": """Problem: {problem}
Step 1: {step_hint}
Answer: """
        }
    
    def optimize_prompt(self, task_type: str, **kwargs) -> str:
        """Create optimized prompt for small models"""
        
        if task_type not in self.prompt_templates:
            return self._create_generic_prompt(**kwargs)
        
        template = self.prompt_templates[task_type]
        
        # Ensure prompt fits in context window
        prompt = template.format(**kwargs)
        
        if self._count_tokens(prompt) > self.context_limit * 0.7:  # Leave room for response
            prompt = self._compress_prompt(prompt, kwargs)
        
        return prompt
    
    def _compress_prompt(self, prompt: str, kwargs: Dict) -> str:
        """Compress prompt to fit context window"""
        
        # Strategy 1: Truncate long inputs
        if "text" in kwargs and len(kwargs["text"]) > 500:
            kwargs["text"] = kwargs["text"][:500] + "..."
        
        if "context" in kwargs and len(kwargs["context"]) > 300:
            kwargs["context"] = kwargs["context"][:300] + "..."
        
        # Strategy 2: Use simpler language
        simplified_prompt = prompt.replace("Categories:", "Types:")
        simplified_prompt = simplified_prompt.replace("Question:", "Q:")
        simplified_prompt = simplified_prompt.replace("Answer:", "A:")
        
        return simplified_prompt
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simplified)"""
        return len(text.split()) * 1.3  # Rough approximation

class SmallModelAgent:
    """Agent optimized for small language models"""
    
    def __init__(self, model_name: str, context_limit: int = 2048):
        self.model_name = model_name
        self.optimizer = SmallModelOptimizer(model_name, context_limit)
        self.task_cache = {}  # Cache common responses
        self.skill_chains = self._build_skill_chains()
        
    def _build_skill_chains(self) -> Dict[str, List[str]]:
        """Build chains of simple skills that combine to complex capabilities"""
        return {
            "research": ["search", "extract", "summarize"],
            "analysis": ["classify", "compare", "conclude"],
            "writing": ["outline", "draft", "refine"],
            "problem_solving": ["understand", "plan", "execute", "verify"]
        }
    
    async def execute_simple_task(self, task_type: str, **params) -> str:
        """Execute simple, atomic tasks"""
        
        # Check cache first
        cache_key = self._generate_cache_key(task_type, params)
        if cache_key in self.task_cache:
            return self.task_cache[cache_key]
        
        # Generate optimized prompt
        prompt = self.optimizer.optimize_prompt(task_type, **params)
        
        # Execute with small model
        result = await self._execute_with_small_model(prompt)
        
        # Cache result
        self.task_cache[cache_key] = result
        
        return result
    
    async def execute_complex_task(self, complex_task: str, **params) -> Dict[str, Any]:
        """Break complex task into simple steps for small model"""
        
        # Decompose complex task
        skill_chain = self._identify_skill_chain(complex_task)
        
        results = {}
        accumulated_context = ""
        
        for i, skill in enumerate(skill_chain):
            # Prepare parameters for this skill
            skill_params = self._prepare_skill_params(skill, params, accumulated_context)
            
            # Execute simple skill
            skill_result = await self.execute_simple_task(skill, **skill_params)
            results[f"step_{i+1}_{skill}"] = skill_result
            
            # Accumulate context for next step
            accumulated_context += f"\n{skill}: {skill_result}"
            
            # Trim context if getting too long
            if len(accumulated_context) > 500:
                accumulated_context = accumulated_context[-400:]  # Keep recent context
        
        # Final synthesis
        final_result = await self._synthesize_results(complex_task, results)
        
        return {
            "final_result": final_result,
            "step_results": results,
            "skill_chain": skill_chain
        }
    
    def _identify_skill_chain(self, complex_task: str) -> List[str]:
        """Identify which skill chain to use for complex task"""
        
        task_lower = complex_task.lower()
        
        if any(word in task_lower for word in ["research", "find", "investigate"]):
            return self.skill_chains["research"]
        elif any(word in task_lower for word in ["analyze", "compare", "evaluate"]):
            return self.skill_chains["analysis"] 
        elif any(word in task_lower for word in ["write", "create", "compose"]):
            return self.skill_chains["writing"]
        elif any(word in task_lower for word in ["solve", "plan", "fix"]):
            return self.skill_chains["problem_solving"]
        else:
            return ["understand", "execute"]  # Default simple chain
    
    def _prepare_skill_params(self, skill: str, original_params: Dict, context: str) -> Dict:
        """Prepare parameters for individual skill execution"""
        
        skill_params = {**original_params}
        
        if skill == "search":
            skill_params["query"] = original_params.get("topic", "")
        elif skill == "extract":
            skill_params["text"] = context
            skill_params["target"] = "key information"
        elif skill == "summarize":
            skill_params["text"] = context
            skill_params["length"] = "50"
        elif skill == "classify":
            skill_params["text"] = original_params.get("content", context)
        
        return skill_params
    
    async def _synthesize_results(self, original_task: str, step_results: Dict) -> str:
        """Synthesize step results into final answer"""
        
        # Create simple synthesis prompt
        results_text = "\n".join([f"{step}: {result}" for step, result in step_results.items()])
        
        synthesis_prompt = f"""Task: {original_task}
Results: {results_text}

Final answer: """
        
        return await self._execute_with_small_model(synthesis_prompt)
    
    async def _execute_with_small_model(self, prompt: str) -> str:
        """Execute prompt with small model (placeholder)"""
        # This would integrate with your actual small model
        # For demo purposes, returning a placeholder
        return f"[Small model response to: {prompt[:50]}...]"

### 10.2 Efficient Tool Use with Small Models

class SmallModelToolAgent:
    """Tool-enabled agent optimized for small models"""
    
    def __init__(self, small_model_agent: SmallModelAgent):
        self.agent = small_model_agent
        self.simple_tools = self._create_simple_tools()
        self.tool_selection_cache = {}
        
    def _create_simple_tools(self) -> Dict[str, Callable]:
        """Create simple, focused tools for small models"""
        return {
            "calculator": self._simple_calculator,
            "search": self._simple_search,
            "extract_numbers": self._extract_numbers,
            "extract_dates": self._extract_dates,
            "count_words": self._count_words,
            "format_text": self._format_text
        }
    
    def _simple_calculator(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            # Only allow basic math operations for safety
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return str(result)
            else:
                return "Error: Invalid expression"
        except:
            return "Error: Cannot calculate"
    
    def _simple_search(self, query: str) -> str:
        """Simple search tool (mock)"""
        return f"Search results for '{query}': [Mock result 1, Mock result 2]"
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        import re
        # Simple date patterns
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        return re.findall(date_pattern, text)
    
    def _count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def _format_text(self, text: str, format_type: str) -> str:
        """Format text in various ways"""
        if format_type == "uppercase":
            return text.upper()
        elif format_type == "lowercase":
            return text.lower()
        elif format_type == "title":
            return text.title()
        else:
            return text
    
    async def solve_with_tools(self, problem: str) -> Dict[str, Any]:
        """Solve problem using tools and small model reasoning"""
        
        # Step 1: Identify needed tools
        needed_tools = await self._identify_needed_tools(problem)
        
        # Step 2: Execute tools in sequence
        tool_results = {}
        for tool_name in needed_tools:
            if tool_name in self.simple_tools:
                # Use small model to determine tool parameters
                params = await self._determine_tool_params(tool_name, problem, tool_results)
                
                # Execute tool
                tool_result = self.simple_tools[tool_name](**params)
                tool_results[tool_name] = tool_result
        
        # Step 3: Use small model to reason about tool results
        final_answer = await self._reason_about_results(problem, tool_results)
        
        return {
            "answer": final_answer,
            "tools_used": list(tool_results.keys()),
            "tool_results": tool_results
        }
    
    async def _identify_needed_tools(self, problem: str) -> List[str]:
        """Identify which tools are needed for this problem"""
        
        # Use cached tool selection if available
        cache_key = self._hash_problem(problem)
        if cache_key in self.tool_selection_cache:
            return self.tool_selection_cache[cache_key]
        
        # Simple keyword-based tool selection (optimized for small models)
        needed_tools = []
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["calculate", "math", "sum", "multiply", "divide"]):
            needed_tools.append("calculator")
        
        if any(word in problem_lower for word in ["search", "find", "look up"]):
            needed_tools.append("search")
        
        if any(word in problem_lower for word in ["number", "count", "amount"]):
            needed_tools.append("extract_numbers")
        
        if any(word in problem_lower for word in ["date", "when", "time"]):
            needed_tools.append("extract_dates")
        
        if any(word in problem_lower for word in ["words", "length", "size"]):
            needed_tools.append("count_words")
        
        # Cache the selection
        self.tool_selection_cache[cache_key] = needed_tools
        
        return needed_tools
    
    async def _determine_tool_params(self, tool_name: str, problem: str, previous_results: Dict) -> Dict:
        """Determine parameters for tool execution"""
        
        if tool_name == "calculator":
            # Extract mathematical expression
            return {"expression": await self._extract_math_expression(problem)}
        
        elif tool_name == "search":
            # Extract search query
            return {"query": await self._extract_search_query(problem)}
        
        elif tool_name in ["extract_numbers", "extract_dates", "count_words"]:
            # These tools work on text
            return {"text": problem}
        
        elif tool_name == "format_text":
            return {
                "text": problem,
                "format_type": await self._determine_format_type(problem)
            }
        
        else:
            return {}
    
    async def _extract_math_expression(self, problem: str) -> str:
        """Extract mathematical expression from problem"""
        
        # Use small model with simple prompt
        prompt = f"""Extract the math from: {problem}
Math: """
        
        result = await self.agent._execute_with_small_model(prompt)
        return result.strip()
    
    def _hash_problem(self, problem: str) -> str:
        """Create hash for problem caching"""
        import hashlib
        return hashlib.md5(problem.encode()).hexdigest()[:8]

### 10.3 Specialized Small Model Patterns

class SmallModelSpecialist:
    """Specialized agents that maximize small model capabilities in specific domains"""
    
    def __init__(self, domain: str, model_agent: SmallModelAgent):
        self.domain = domain
        self.agent = model_agent
        self.domain_templates = self._load_domain_templates()
        self.domain_keywords = self._load_domain_keywords()
        
    def _load_domain_templates(self) -> Dict[str, str]:
        """Load domain-specific optimized templates"""
        
        templates = {
            "customer_service": {
                "classify_intent": "Customer says: {message}\nIntent: ",
                "generate_response": "Issue: {issue}\nResponse: ",
                "escalate_check": "Problem: {problem}\nEscalate? "
            },
            
            "data_entry": {
                "extract_fields": "Text: {text}\nFields: {fields}\nValues: ",
                "validate_data": "Data: {data}\nValid? ",
                "format_output": "Input: {input}\nFormat: {format}\nOutput: "
            },
            
            "content_moderation": {
                "check_safety": "Content: {content}\nSafe? ",
                "categorize_violation": "Content: {content}\nViolation type: ",
                "suggest_action": "Issue: {issue}\nAction: "
            }
        }
        
        return templates.get(self.domain, {})
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords for optimization"""
        
        keywords = {
            "customer_service": ["complaint", "refund", "billing", "support", "help"],
            "data_entry": ["name", "address", "phone", "email", "date"],
            "content_moderation": ["spam", "inappropriate", "violation", "policy"]
        }
        
        return keywords.get(self.domain, [])
    
    async def process_domain_task(self, task_type: str, **params) -> str:
        """Process task using domain-specific optimizations"""
        
        if task_type in self.domain_templates:
            # Use domain-specific template
            template = self.domain_templates[task_type]
            prompt = template.format(**params)
        else:
            # Fall back to general approach
            prompt = f"Task: {task_type}\nInput: {params}\nOutput: "
        
        # Add domain context if helpful
        if len(prompt) < 1000:  # Only if we have room
            domain_context = f"Domain: {self.domain}. "
            prompt = domain_context + prompt
        
        return await self.agent._execute_with_small_model(prompt)
    
    async def batch_process(self, tasks: List[Dict]) -> List[str]:
        """Process multiple similar tasks efficiently"""
        
        results = []
        
        # Group similar tasks
        grouped_tasks = self._group_similar_tasks(tasks)
        
        for task_type, task_group in grouped_tasks.items():
            # Process each group with optimized batching
            group_results = await self._process_task_group(task_type, task_group)
            results.extend(group_results)
        
        return results
    
    def _group_similar_tasks(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar tasks for batch processing"""
        
        grouped = {}
        for task in tasks:
            task_type = task.get("type", "unknown")
            if task_type not in grouped:
                grouped[task_type] = []
            grouped[task_type].append(task)
        
        return grouped

## Learning Path and Advanced Resources

### Structured Learning Progression

The journey from prompt engineering basics to production AI systems requires a structured approach. Here's a comprehensive learning path that incorporates all the advanced techniques covered in this guide.

#### Foundation Phase (Weeks 1-4)

**Week 1-2: Prompt Engineering Mastery**
- Master basic prompt engineering techniques
- Practice Chain-of-Thought and Tree-of-Thought prompting
- Implement self-consistency and few-shot learning
- Build dynamic example selection systems

**Practical Projects:**
```python
# Week 1 Project: Advanced Classification System
class AdvancedClassifier:
    def __init__(self):
        self.few_shot_prompter = DynamicFewShotPrompting(example_bank)
        self.consistency_prompter = SelfConsistencyPrompting(llm)
    
    async def classify_with_confidence(self, text: str) -> Dict[str, Any]:
        # Use few-shot learning
        few_shot_prompt = self.few_shot_prompter.create_few_shot_prompt(
            text, "Classify sentiment as positive, negative, or neutral"
        )
        
        # Apply self-consistency
        result = await self.consistency_prompter.generate_multiple_responses(
            few_shot_prompt, num_samples=5
        )
        
        consistent_answer = self.consistency_prompter.find_consistent_answer(result)
        
        return {
            "classification": consistent_answer["final_answer"],
            "confidence": consistent_answer["confidence"],
            "reasoning_paths": result
        }
```

**Week 3-4: Context Engineering**
- Understand context window optimization
- Implement hierarchical context management
- Build advanced memory systems
- Practice information prioritization

**Practical Projects:**
```python
# Week 3 Project: Intelligent Document Assistant
class DocumentAssistant:
    def __init__(self):
        self.context_manager = HierarchicalContextManager()
        self.memory_system = AdvancedMemorySystem(llm, vector_store)
    
    async def answer_document_question(self, question: str, documents: List[str]) -> str:
        # Process documents with hierarchical context
        for doc in documents:
            self.context_manager.add_context("retrieved_knowledge", doc)
        
        # Get relevant memories
        relevant_memories = self.memory_system.get_relevant_memories(question)
        self.context_manager.add_context("conversation_history", relevant_memories)
        
        # Build optimized context
        context = self.context_manager.build_optimized_context()
        
        # Generate response
        response = await llm.generate(f"{context}\n\nQuestion: {question}\nAnswer:")
        
        # Update memory
        self.memory_system.update_memory({
            "query": question,
            "response": response,
            "context": context
        })
        
        return response
```

#### Intermediate Phase (Weeks 5-12)

**Week 5-6: Basic RAG Implementation**
- Implement simple RAG systems
- Practice document chunking and embedding
- Build basic retrieval mechanisms
- Integrate with vector databases

**Week 7-8: Advanced RAG Architectures**
- Implement Hierarchical RAG (HRAG)
- Build Graph RAG systems
- Create Multimodal RAG
- Develop Agentic RAG with self-reflection

**Practical Projects:**
```python
# Week 7 Project: Multi-Modal Knowledge System
class MultiModalKnowledgeSystem:
    def __init__(self):
        self.hierarchical_rag = HierarchicalRAG(llm, embeddings)
        self.multimodal_rag = MultimodalRAG(llm, text_embeddings)
        self.agentic_rag = AgenticRAG(llm, retriever)
    
    async def comprehensive_search(self, query: str, include_multimodal: bool = True) -> Dict[str, Any]:
        # Hierarchical retrieval
        hierarchical_results = await self.hierarchical_rag.answer_with_hierarchy(query)
        
        # Multimodal search if requested
        multimodal_results = None
        if include_multimodal:
            multimodal_results = await self.multimodal_rag.multimodal_search(query)
        
        # Self-reflective search
        agentic_results = await self.agentic_rag.self_reflective_retrieval(query)
        
        # Combine and synthesize results
        return await self._synthesize_results(query, {
            "hierarchical": hierarchical_results,
            "multimodal": multimodal_results,
            "agentic": agentic_results
        })
```

**Week 9-10: MCP and Tool Integration**
- Implement MCP servers and clients
- Build advanced tool use patterns
- Create tool chaining workflows
- Practice error handling and fallbacks

**Week 11-12: Single Agent Systems**
- Build reflective agents
- Implement planning agents with decomposition
- Create advanced tool use agents
- Develop learning and adaptation mechanisms

#### Advanced Phase (Weeks 13-20)

**Week 13-14: Multi-Agent Coordination**
- Implement message brokers and communication
- Build hierarchical agent systems
- Create debate-based decision making
- Develop consensus building mechanisms

**Week 15-16: Production Systems**
- Implement comprehensive monitoring
- Build safety and ethics systems
- Create performance optimization
- Develop error handling and resilience

**Week 17-18: Hybrid Model Strategies**
- Implement knowledge distillation
- Build model routing systems
- Create performance benchmarking
- Develop continuous improvement loops

**Week 19-20: Advanced Integration**
- Combine all techniques into cohesive systems
- Build domain-specific applications
- Implement real-world use cases
- Create scalable deployment strategies

### Practical Project Recommendations

#### Progressive Project Complexity

**Project 1: Personal Knowledge Assistant (Weeks 1-4)**
Build a personal assistant that can answer questions about your documents and remember conversations.

```python
class PersonalKnowledgeAssistant:
    def __init__(self):
        self.rag_system = SimpleRAGSystem()
        self.memory_system = ConversationMemory()
        self.context_manager = ContextManager()
        
    async def chat(self, user_input: str) -> str:
        # Add user message to memory
        self.memory_system.add_message("user", user_input)
        
        # Search knowledge base
        relevant_docs = self.rag_system.search(user_input)
        
        # Build context
        context = self.context_manager.optimize_context(relevant_docs, user_input)
        conversation_context = self.memory_system.get_context_for_prompt()
        
        # Generate response
        prompt = f"""
        {conversation_context}
        
        Knowledge base context:
        {context}
        
        User: {user_input}
        Assistant: """
        
        response = await llm.generate(prompt)
        
        # Update memory
        self.memory_system.add_message("assistant", response)
        
        return response
```

**Project 2: Multi-Modal Document Analyzer (Weeks 5-8)**
Create a system that can analyze documents containing text, images, and tables.

```python
class DocumentAnalyzer:
    def __init__(self):
        self.multimodal_rag = MultimodalRAG(llm, text_embeddings)
        self.hierarchical_rag = HierarchicalRAG(llm, embeddings)
        
    async def analyze_document(self, document_path: str, analysis_request: str) -> Dict[str, Any]:
        # Process multimodal content
        multimodal_content = self.multimodal_rag.process_multimodal_document(document_path)
        
        # Build hierarchical index
        self.hierarchical_rag.build_hierarchical_index([multimodal_content])
        
        # Perform analysis
        text_analysis = await self._analyze_text_content(multimodal_content["text"], analysis_request)
        image_analysis = await self._analyze_image_content(multimodal_content["images"], analysis_request)
        table_analysis = await self._analyze_table_content(multimodal_content["tables"], analysis_request)
        
        # Synthesize results
        return {
            "document_path": document_path,
            "analysis_request": analysis_request,
            "text_insights": text_analysis,
            "visual_insights": image_analysis,
            "data_insights": table_analysis,
            "integrated_analysis": await self._integrate_multimodal_analysis(
                text_analysis, image_analysis, table_analysis, analysis_request
            )
        }
```

**Project 3: Collaborative Research Agent (Weeks 9-12)**
Build a multi-agent system for comprehensive research tasks.

```python
class CollaborativeResearchSystem:
    def __init__(self):
        self.broker = MessageBroker()
        self.hierarchical_system = HierarchicalMultiAgentSystem(self.broker)
        self.debate_system = DebateMultiAgentSystem(self.broker)
        
    async def conduct_research(self, research_topic: str, research_depth: str = "comprehensive") -> Dict[str, Any]:
        # Phase 1: Initial research by hierarchical system
        research_results = await self.hierarchical_system.execute_collaborative_task(
            f"Conduct {research_depth} research on: {research_topic}",
            {"depth": research_depth, "deliverable": "research report"}
        )
        
        # Phase 2: Critical evaluation through debate
        debate_results = await self.debate_system.debate_resolution(
            f"Evaluate the quality and completeness of research on: {research_topic}",
            rounds=3,
            include_evidence_analysis=True
        )
        
        # Phase 3: Final synthesis
        final_report = await self._synthesize_research_and_debate(
            research_topic, research_results, debate_results
        )
        
        return {
            "topic": research_topic,
            "research_phase": research_results,
            "evaluation_phase": debate_results,
            "final_report": final_report,
            "quality_score": self._assess_research_quality(final_report)
        }
```

**Project 4: Enterprise Knowledge System (Weeks 13-20)**
Deploy a production-ready system with full monitoring and safety features.

```python
class EnterpriseKnowledgeSystem:
    def __init__(self):
        # Core systems
        self.hybrid_model_system = HybridModelSystem(models, rag_system)
        self.safety_system = AdvancedSafetySystem(llm)
        self.monitor = LLMApplicationMonitor()
        
        # Multi-agent coordination
        self.agent_system = HierarchicalMultiAgentSystem(MessageBroker())
        
        # Production features
        self.distillation_system = AdvancedKnowledgeDistillation(teacher_model, student_model)
        
    async def process_enterprise_query(self, query: str, user_context: Dict, 
                                     safety_level: str = "high") -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            # Safety check on input
            input_safety = self.safety_system.comprehensive_safety_check("", query, user_context)
            if not input_safety["safe"] and safety_level == "high":
                return {
                    "error": "Query blocked by safety filters",
                    "safety_issues": input_safety["warnings"]
                }
            
            # Route to optimal processing
            result = await self.hybrid_model_system.intelligent_generation(query, user_context)
            
            # Safety check on output
            output_safety = self.safety_system.comprehensive_safety_check(
                query, result["response"], user_context
            )
            
            if not output_safety["safe"]:
                # Apply content moderation
                moderated_response = self.safety_system.adaptive_content_moderation(
                    result["response"], user_context
                )
                result["response"] = moderated_response
                result["moderated"] = True
            
            # Quality evaluation
            quality_metrics = self.monitor.evaluate_response_quality(
                query, result["response"], context=user_context.get("expected_response")
            )
            
            # Performance monitoring
            processing_time = time.time() - start_time
            self.monitor.metrics.latency.append(processing_time)
            
            return {
                "response": result["response"],
                "routing_info": result["routing_decision"],
                "quality_metrics": quality_metrics,
                "safety_checks": {
                    "input_safety": input_safety,
                    "output_safety": output_safety
                },
                "performance": {
                    "processing_time": processing_time,
                    "model_used": result["model_used"]
                },
                "success": True
            }
            
        except Exception as e:
            # Error handling and logging
            error_result = {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
            
            # Update error metrics
            self.monitor.metrics.error_rate += 1
            
            return error_result
    
    async def continuous_improvement(self):
        """Continuous model improvement through distillation and feedback"""
        
        # Collect performance data
        performance_data = self.monitor.get_real_time_metrics()
        
        # If performance is below threshold, trigger improvement
        if performance_data["current_avg_quality"] < 0.7:
            # Generate training queries from recent low-quality interactions
            training_queries = await self._extract_training_queries()
            
            # Run knowledge distillation
            distillation_results = await self.distillation_system.progressive_distillation(
                training_queries, iterations=3
            )
            
            # Update routing based on results
            await self._update_routing_strategy(distillation_results)
        
        return {
            "improvement_triggered": performance_data["current_avg_quality"] < 0.7,
            "current_performance": performance_data
        }
```

### Learning Resources and Best Practices

#### Recommended Reading and Research

**Research Papers:**
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "Constitutional AI: Harmlessness from AI Feedback"
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

**Technical Documentation:**
- LangChain Documentation: Advanced patterns and implementations
- OpenAI API Best Practices: Production deployment guidelines
- Anthropic Safety Research: Responsible AI development
- Hugging Face Transformers: Model fine-tuning and optimization

#### Community and Collaboration

**Open Source Contributions:**
- Contribute to LangChain framework improvements
- Develop reusable prompt templates and patterns
- Create domain-specific agent implementations
- Share performance optimization techniques

**Professional Development:**
- Join AI/ML communities and forums
- Participate in research paper discussions
- Attend conferences and workshops
- Build a portfolio of increasingly complex projects

#### Advanced Techniques Mastery Checklist

**Level 1 Mastery: Prompt Engineering**
- [ ] Can write effective prompts for any task type
- [ ] Understands when to use different prompting techniques
- [ ] Can implement dynamic example selection
- [ ] Masters chain-of-thought and tree-of-thought reasoning

**Level 2 Mastery: Context Engineering**
- [ ] Can optimize context for any token limit
- [ ] Implements hierarchical information organization
- [ ] Builds sophisticated memory systems
- [ ] Masters information prioritization algorithms

**Level 3 Mastery: RAG Systems**
- [ ] Can implement any RAG architecture variant
- [ ] Understands retrieval quality optimization
- [ ] Masters multimodal content integration
- [ ] Implements self-improving retrieval systems

**Level 4 Mastery: Agent Systems**
- [ ] Builds autonomous agents with planning capabilities
- [ ] Implements multi-agent coordination protocols
- [ ] Masters tool use and integration patterns
- [ ] Creates learning and adaptation mechanisms

**Level 5 Mastery: Production Systems**
- [ ] Deploys scalable, monitored AI applications
- [ ] Implements comprehensive safety and ethics systems
- [ ] Masters performance optimization techniques
- [ ] Creates continuous improvement pipelines

---

## Conclusion and Future Directions

This comprehensive guide has taken you through the complete journey from basic prompt engineering to building sophisticated, production-ready AI systems. The progression through each level provides not just technical knowledge, but practical experience with increasingly complex real-world applications.

### Key Achievements Through This Guide

**Technical Mastery:**
You've learned to implement every major pattern in modern LLM applications, from basic prompts to multi-agent systems with production-grade monitoring and safety features.

**Architectural Understanding:**
You understand how to design systems that scale from simple chatbots to enterprise knowledge management platforms, with proper consideration for performance, safety, and user experience.

**Best Practices Integration:**
You've learned not just what to build, but how to build it responsibly, with appropriate safety measures, monitoring systems, and continuous improvement mechanisms.

### The Future of LLM Applications

**Emerging Trends:**
- **Multimodal Integration:** Systems that seamlessly work with text, images, audio, and video
- **Agent Ecosystems:** Large networks of specialized agents collaborating on complex tasks
- **Adaptive Learning:** Systems that continuously improve from user interactions
- **Edge Deployment:** Efficient models running locally for privacy and speed

**Technical Frontiers:**
- **Advanced Reasoning:** More sophisticated logical and mathematical reasoning capabilities
- **Long-term Memory:** Systems that maintain coherent memory across extended interactions
- **Tool Creation:** Agents that can create and modify their own tools
- **Self-Modification:** Systems that can improve their own architecture and capabilities

### Continuing Your Journey

**Immediate Next Steps:**
1. Choose a domain-specific application to focus on
2. Build a complete system using the patterns learned
3. Deploy to production with proper monitoring
4. Gather user feedback and iterate

**Long-term Development:**
1. Contribute to open-source LLM frameworks
2. Research novel applications in your domain
3. Develop new patterns and share with the community
4. Mentor others beginning their LLM journey

**Staying Current:**
- Follow research developments in major AI labs
- Participate in the open-source community
- Experiment with new models and techniques
- Build connections with other practitioners

### Final Thoughts

The field of LLM applications is rapidly evolving, with new capabilities and techniques emerging constantly. The patterns and principles you've learned in this guide provide a solid foundation, but the most important skill is the ability to adapt and learn continuously.

Remember that building AI systems is not just about technical capability—it's about creating tools that genuinely help people solve real problems. As you apply these techniques, always consider the human impact of your work and strive to build systems that are not only powerful but also safe, fair, and beneficial.

The future of AI applications lies in the hands of practitioners like you who understand both the technical depth and the responsible application of these powerful technologies. Use your knowledge wisely, and help shape a future where AI systems truly serve humanity's best interests.

**The journey from prompt to production is now complete. The journey of innovation has just begun.**

---

*This guide represents the current state of LLM application development as of 2025. As the field continues to evolve rapidly, remember to stay curious, keep learning, and always prioritize responsible development practices.*
