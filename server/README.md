# 🏗️ BBMP Road Repair Prioritization Environment

An OpenEnv environment where an AI agent learns to prioritize road repair
complaints in Bengaluru like a BBMP municipal officer.

## 🌍 Real World Problem

Bengaluru has 13,000+ km of roads. BBMP receives 400+ complaints daily.
Engineers currently prioritize manually using gut feel — no optimization,
no data-driven decisions. This leads to:
- Critical roads near schools and hospitals ignored
- Poor residential areas waiting 6+ months for repairs
- 50+ pothole deaths per year in Bengaluru alone

## 🤖 What the Agent Does

The agent acts as a BBMP officer. Every step it sees:
- Active road complaints across Bengaluru wards
- Remaining budget and repair crews
- Weather conditions (rain creates new damage)

It decides: which road to fix, what type of repair, in what order.

## 📋 Tasks

| Task | Difficulty | Wards | Complaints | Budget |
|------|-----------|-------|------------|--------|
| task1 | Easy | 1 | 5 | ₹1 lakh |
| task2 | Medium | 3 | 15 | ₹3 lakh |
| task3 | Hard | 5 | 30 | ₹5 lakh |

## 🎯 Reward Function

| Factor | Reward |
|--------|--------|
| Fix critical road | +10 to +20 |
| Fix school/hospital zone | 1.8x multiplier |
| Road pending 7+ days | +0.5 per extra day |
| High traffic impact | +up to 5.0 |
| Patch on low severity | +2.0 efficiency bonus |
| Leave critical unresolved | -15 penalty |
| Wait action | -2 penalty |

## 🚀 Quick Start
```bash
pip install fastapi uvicorn pydantic openenv-core openai
cd server
uvicorn app:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/reset?task_id=task1
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"repair","complaint_id":"KOR-001","repair_type":"patch"}'
```

## 📊 Baseline Scores

| Task | Score |
|------|-------|
| task1 | 0.9959 |
| task2 | 0.5512 |
| task3 | 0.4725 |

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | HuggingFace API key |

## 📁 Structure