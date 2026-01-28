# Assignment 5 – Using TOPSIS to Pick a Pre-Trained Text Generation Model (Like a Normal Human Would)

## 1. What are we even doing?

The question for this assignment is very practical:

> "I want to use a pre-trained model for text generation. There are so many options.  
>  Which one should I pick if I care about quality **and** speed **and** not burning my GPU?"

Instead of choosing randomly, we treat this as a **multi-criteria decision problem** and use  
**TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** to rank different models.

The idea is simple:

- Define a few **criteria** that matter in real life,
- Give each criterion a **weight**,
- Let TOPSIS tell us which model is **closest to the “ideal”** one (best possible on all criteria).

---

## 2. Models Compared (9 total, all free / open-weight)

We compare nine popular, freely available models that you can download from Hugging Face / EleutherAI:

1. **DistilGPT-2** – `distilgpt2` (~82M parameters)  
2. **GPT-2 Small** – `gpt2` (~117M parameters)  
3. **GPT-Neo 125M** – `EleutherAI/gpt-neo-125M` (125M)  
4. **GPT-2 Medium** – `gpt2-medium` (~345M parameters)  
5. **GPT-2 Large** – `gpt2-large` (~774M parameters)  
6. **GPT-2 XL** – `gpt2-xl` (~1.5B parameters)  
7. **GPT-Neo 1.3B** – `EleutherAI/gpt-neo-1.3B` (1.3B)  
8. **GPT-J 6B** – `EleutherAI/gpt-j-6B` (6B)  
9. **Mistral 7B** – `mistralai/Mistral-7B-v0.1` (7B)

Some of these are "laptop-friendly"; others basically scream "please give me an A100".

---

## 3. Criteria: What matters and how much?

We look at five criteria that a student / practitioner actually cares about:

### 3.1 C1 – Text Quality (Benefit)

- Rough 0–100 score representing how strong the model is at generation (fluency, coherence, general capability).
- Bigger models are generally better, but with **diminishing returns**.
- We model this as a function of the logarithm of parameter count.

### 3.2 C2 – Model Size (Million Parameters, Cost)

- Total number of parameters in **millions**.
- This directly impacts download time, storage, and sometimes load time.
- In TOPSIS, this is a **cost**: smaller is better.

### 3.3 C3 – Inference Speed (tokens/second, Benefit)

- Approximate speed for generating tokens.
- We assume speed is **inversely proportional** to parameter count:
  larger models → slower.

### 3.4 C4 – GPU Memory Usage (GB, Cost)

- How much VRAM does it roughly take to load the model?
- We use a simple model:
  \[
  \text{Memory\_GB} \approx 1 + 0.002 \times \text{Params\_M}
  \]
  (e.g., 500M → ~2 GB, 6000M → ~13 GB)

### 3.5 C5 – Ecosystem & Fine-Tuning Support (Benefit)

- How easy is it to actually work with the model in the real world?
- Things we consider:
  - Number of tutorials, notebooks, examples,
  - Pre-made LoRA adapters / checkpoints,
  - Community usage and support.
- Scored on a **1–10** scale (subjective, but justified in context).

---

## 4. Weights and Impacts

### 4.1 Weights

We assign the following weights (sum = 1.0):

- **w₁ (C1: Quality)** = 0.35  
- **w₂ (C2: Params)** = 0.10  
- **w₃ (C3: Speed)** = 0.25  
- **w₄ (C4: Memory)** = 0.15  
- **w₅ (C5: Ecosystem)** = 0.15  

Interpretation:

- Quality and speed are the main heroes,
- VRAM and ecosystem also matter a lot,
- Raw parameter count alone matters a little less.

### 4.2 Impacts

- **C1: + (benefit)** – higher quality is better  
- **C2: − (cost)** – fewer parameters is better  
- **C3: + (benefit)** – higher speed is better  
- **C4: − (cost)** – lower memory usage is better  
- **C5: + (benefit)** – richer ecosystem is better  

---

## 5. Decision Matrix

The concrete values that go into TOPSIS are:

| Model         | C1: Quality ↑ | C2: Params (M) ↓ | C3: Speed (tok/s) ↑ | C4: Memory (GB) ↓ | C5: Ecosystem ↑ |
|---------------|---------------|------------------|---------------------|-------------------|-----------------|
| DistilGPT-2   | 56.84         | 82               | 609.8               | 1.16              | 9               |
| GPT-2 Small   | 60.00         | 117              | 427.4               | 1.23              | 10              |
| GPT-Neo 125M  | 60.59         | 125              | 400.0               | 1.25              | 8               |
| GPT-2 Medium  | 69.61         | 345              | 144.9               | 1.69              | 9               |
| GPT-2 Large   | 76.80         | 774              | 64.6                | 2.55              | 8               |
| GPT-2 XL      | 82.68         | 1500             | 33.3                | 4.00              | 7               |
| GPT-Neo 1.3B  | 81.40         | 1300             | 38.5                | 3.60              | 7               |
| GPT-J 6B      | 95.00         | 6000             | 8.3                 | 13.00             | 8               |
| Mistral 7B    | 96.37         | 7000             | 7.1                 | 15.00             | 9               |

> **Important:**  
> These are **approximations** meant to demonstrate the decision-making process.  
> A full research paper would use measured benchmarks, not just formula-based estimates.

---

## 6. TOPSIS Methodology (Step-by-Step)

All these steps are implemented in the accompanying Colab notebook.

1. **Construct the decision matrix** \( X = [x_{ij}] \) with 9 models × 5 criteria.

2. **Normalize the matrix** using vector normalization:
   \[
   r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}
   \]

3. **Apply weights** to get the weighted normalized matrix:
   \[
   v_{ij} = w_j \cdot r_{ij}
   \]

4. **Find the ideal best (A⁺) and ideal worst (A⁻)**:
   - For **benefit** criteria:
     \[
     v_j^+ = \max_i v_{ij}, \quad v_j^- = \min_i v_{ij}
     \]
   - For **cost** criteria:
     \[
     v_j^+ = \min_i v_{ij}, \quad v_j^- = \max_i v_{ij}
     \]

5. **Compute distances** of each model to A⁺ and A⁻:
   \[
   S_i^+ = \sqrt{\sum_{j} (v_{ij} - v_j^+)^2}, \quad
   S_i^- = \sqrt{\sum_{j} (v_{ij} - v_j^-)^2}
   \]

6. **Compute the closeness coefficient**:
   \[
   C_i = \frac{S_i^-}{S_i^+ + S_i^-}
   \]
   Higher \(C_i\) ⇒ closer to the ideal solution ⇒ better.

7. **Rank the models** by decreasing \(C_i\).

---

## 7. Results

### 7.1 Numerical Results

The final TOPSIS scores are:

| Rank | Model         | S⁺ (→ ideal best) | S⁻ (→ ideal worst) | Closeness Cᵢ   |  
|------|---------------|-------------------|---------------------|---------------|
| 1    | DistilGPT-2   | 0.0604            | 0.2143              | **0.7802**    |
| 2    | GPT-2 Small   | 0.0765            | 0.1739              | **0.6944**    |
| 3    | GPT-Neo 125M  | 0.0825            | 0.1675              | **0.6701**    |
| 4    | GPT-2 Medium  | 0.1411            | 0.1272              | 0.4741        |
| 5    | GPT-2 Large   | 0.1619            | 0.1164              | 0.4182        |
| 6    | GPT-Neo 1.3B  | 0.1697            | 0.1086              | 0.3903        |
| 7    | GPT-2 XL      | 0.1714            | 0.1059              | 0.3818        |
| 8    | GPT-J 6B      | 0.2043            | 0.0610              | 0.2299        |
| 9    | Mistral 7B    | 0.2140            | 0.0613              | 0.2225        |

### 7.2 Result Graph

The notebook also plots a **bar chart** with:

- **x-axis:** Model name  
- **y-axis:** Closeness coefficient \(C_i\)

Visually, you see three clear “tiers”:

1. **Top tier:** DistilGPT-2, GPT-2 Small, GPT-Neo 125M  
2. **Middle tier:** GPT-2 Medium, GPT-2 Large, GPT-Neo 1.3B, GPT-2 XL  
3. **Heavy monsters tier:** GPT-J 6B, Mistral 7B

![alt text](image.png)

---

## 8. Discussion (in normal language)

If you look at the raw quality scores alone, the big boys (GPT-J, Mistral) win easily.  
But reality isn’t that simple:

- They are **slow**,  
- They require **huge VRAM**,  
- And they are often **overkill** for everyday use (especially in a college setting).

Under the chosen weights, TOPSIS is basically answering:

> “Given I care about speed, memory, and ease of use, not just bragging rights,  
> what is the most reasonable model to pick?”

The answer is:

- **1st: DistilGPT-2** – tiny, fast, widely supported, quality is decent.  
- **2nd & 3rd: GPT-2 Small and GPT-Neo 125M** – still easy to run, slightly bigger and slower, but stronger.

In other words:

> For a normal developer / student with limited GPU,  
> **DistilGPT-2, GPT-2 Small, and GPT-Neo 125M** give the best *overall* trade-off.

---

## 9. Limitations & Future Work

- Quality, speed, memory and ecosystem scores are **not measured benchmarks** – they are reasonable approximations.
- A more rigorous study would:
  - Benchmark each model on standard datasets and compute real metrics (perplexity, BLEU, etc.),
  - Measure actual tokens/sec and peak memory on the same GPU,
  - Possibly add more criteria like energy usage, license constraints, etc.

But for this assignment, the goal is to clearly demonstrate:

- How to model model-selection as a **multi-criteria decision problem**, and  
- How **TOPSIS** can give a sensible ranking given your priorities.

---

## THANK YOU!!

