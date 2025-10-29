# Conservative Q-Learning for Optimal Hint Timing in LLM-Driven Computer Science Education

---

## Overview
This project investigates whether a **fully offline reinforcement learning (RL) agent** can learn effective hint timing policies for programming education.

- Objective: Learn when to offer hints that maximise student understanding.
- Approach: Offline Conservative Q-Learning (CQL) trained on historical tutor logs (iSnap dataset).
- Reward Signal Derived from an Elo-style mastery estimate updated per student attempt.
- Integration A GPT-4o model generates mastery-conditioned, accessibility-constrained hints.

---

## Key Features
- Offline RL Training: CQL agent trained on preprocessed replay buffer of student–problem interactions.
- Mastery Estimation: Elo-based reward function (`Δθ`) reflecting changes in predicted understanding.
- Preprocessing: Pipeline Temporal sorting, one-hot encoding of problems, cumulative hint tracking, and β-difficulty estimation.
- Baselines: Always-hint, never-hint, θ-threshold heuristic, vanilla DQN for comparison.
- Evaluation: Offline policy value estimated using Fitted Q Evaluation (FQE).
- LLM Hint Generation: GPT-4o prompted with problem context and predicted mastery; supports multimodal and dyslexia friendly output.

---

## Dependencies
- Python 3.10+
- `numpy`, `pandas`, `sklearn`, `torch`
- `matplotlib` (for visualisation)

---

## Usage
1. **Prepare data:**  
   Place the iSnap `.txt` dataset (not included) in the `data/` directory.

2. **Preprocess:**  
   Run the preprocessing cells to compute β (difficulty) and θ (mastery) columns.

3. **Train:**  
   Execute the CQL training section. Hyperparameters are configurable at the top of the notebook.

4. **Evaluate:**  
   Compare policy performance using FQE and baselines.

5. **Generate Hints:**  
   Provide a problem name and mastery score to call GPT-4o for a mastery-conditioned hint.

---

## Results Summary
- CQL mean return (FQE): 0.00316
- Vanilla DQN: 0.00178
- Always/never-hint baselines underperform (negative or near-zero reward)
- Removing mastery estimate θ reduces policy value by ~16%
- Accessibility-constrained GPT prompts yield shorter, more structured hints (Flesch–Kincaid ↑ 20%)

---
