# Local LLM Positional Bias Evaluation

This project evaluates positional bias in locally-run Large Language Models (LLMs) using multiple choice questions. The goal is to determine whether LLMs show systematic preferences for certain answer positions (A, B, C, or D) regardless of content.

## 🎯 What is Positional Bias?

Positional bias occurs when a model consistently favors certain positions in multiple choice questions, independent of the actual content. For example, a model might prefer option "A" or "C" more often than would be expected by chance.

## 🛠️ Setup

### Prerequisites
- Windows 10/11 (or adapt for other OS)
- Python 3.8+
- At least 4GB free RAM (for small models)
- Internet connection for initial model download

### Installation Steps

1. **Clone/Download this project**
   ```bash
   cd C:\Users\YourName\Desktop\FYP\test
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama and download models**
   ```bash
   python setup_ollama.py
   ```
   
   This script will:
   - Check if Ollama is installed (if not, provides installation instructions)
   - Start the Ollama service
   - Help you download recommended models for testing

### Recommended Models

- **llama3.2:3b** - Small, fast model (~2GB)
- **mistral:7b** - Balanced performance/size (~4GB)  
- **phi3:mini** - Very lightweight Microsoft model (~2GB)
- **qwen2.5:7b** - Alternative architecture for comparison (~4GB)

## 🧪 Running the Evaluation

### Basic Usage

```bash
python eval_positional_bias.py --model llama3.2:3b --input data/sample_mcq.csv
```

### Full Example with Options

```bash
python eval_positional_bias.py \
    --model mistral:7b \
    --input data/sample_mcq.csv \
    --n-permutations 15 \
    --max-questions 20 \
    --temperature 0.0 \
    --seed 42 \
    --output-prefix results/bias_test
```

### Parameters

- `--model`: Ollama model name (required)
- `--input`: Path to CSV file with questions (default: data/sample_mcq.csv)
- `--n-permutations`: How many times to shuffle each question (default: 10)
- `--max-questions`: Limit number of questions to test (default: all)
- `--temperature`: Model temperature 0.0=deterministic, 1.0=random (default: 0.0)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-prefix`: Where to save results (default: results/positional_bias)

## 📊 Understanding Results

The evaluation produces several key metrics:

### 1. Choice Distribution
Shows how often the model picks A, B, C, or D across all questions:
```
📊 CHOICE DISTRIBUTION (n=200):
   A:   45 (22.5%)
   B:   48 (24.0%)
   C:   62 (31.0%)  ← Potential bias toward C
   D:   45 (22.5%)
```

### 2. Chi-Square Test
Tests if the distribution significantly differs from uniform (25% each):
```
🧮 CHI-SQUARE TEST vs Uniform Distribution:
   Chi-square statistic: 8.420
   P-value: 0.038000
   ✗ Significant deviation from uniform (p < 0.05) - BIAS DETECTED
```

### 3. Position-Dependent Accuracy
Shows accuracy when the correct answer is at each position:
```
🎯 ACCURACY BY CORRECT ANSWER POSITION:
   Overall accuracy: 0.750
   Position-specific accuracy:
     A: 0.780 (n=50, diff=+0.030)
     B: 0.760 (n=50, diff=+0.010)
     C: 0.720 (n=50, diff=-0.030)  ← Worse when correct answer is C
     D: 0.740 (n=50, diff=-0.010)
```

### 4. Position Bias Score
Standard deviation of choice percentages (higher = more biased):
```
📈 POSITION BIAS SCORE: 4.12
   (Standard deviation of choice percentages - higher = more biased)
```

## 📁 File Structure

```
test/
├── eval_positional_bias.py    # Main evaluation script
├── setup_ollama.py           # Ollama setup helper
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data/
│   └── sample_mcq.csv       # Sample questions dataset
└── results/                 # Output directory
    └── positional_bias_*.csv # Detailed results
```

## 📝 Question Format

The CSV should have these columns:
- `id`: Unique question identifier
- `question`: The question text
- `option_a`, `option_b`, `option_c`, `option_d`: The four choices
- `answer`: Correct answer letter (A, B, C, or D)

Example:
```csv
id,question,option_a,option_b,option_c,option_d,answer
q1,What is the capital of France?,Paris,Lyon,Marseille,Toulouse,A
q2,Which planet is red?,Earth,Saturn,Mars,Jupiter,C
```

## 🔬 Methodology

1. **Option Permutation**: For each question, randomly shuffle the positions of A, B, C, D options multiple times
2. **Model Querying**: Ask the model to choose A, B, C, or D for each permutation
3. **Bias Detection**: 
   - Measure if certain positions are chosen more often than others
   - Check if accuracy varies based on where the correct answer is placed
4. **Statistical Testing**: Use chi-square tests to determine statistical significance

## 🚀 Next Steps for Your Research

1. **Expand Dataset**: Test with larger, more diverse question sets
2. **Multiple Models**: Compare bias across different model architectures
3. **Temperature Analysis**: Test how randomness affects bias
4. **Prompt Engineering**: Try different question formats
5. **Control Experiments**: Test with nonsense questions to isolate pure positional preference

## 🛠️ Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama manually
ollama serve
```

### Model Download Issues
```bash
# List available models
ollama list

# Pull specific model
ollama pull llama3.2:3b
```

### Common Errors
- **"Model not found"**: Make sure the model is pulled with `ollama pull model_name`
- **"Connection refused"**: Start Ollama service with `ollama serve`
- **"Out of memory"**: Try smaller models like `phi3:mini` or `llama3.2:3b`

## 📚 Further Reading

- [Ollama Documentation](https://ollama.com)
- Research papers on LLM biases and evaluation methodologies
- Statistical methods for bias detection in AI systems
