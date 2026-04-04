---
title: disaster-response-env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

## Features
- Allocate ambulances to regions
- Severity-based prioritization
- Reward-based evaluation

## Tasks
- Easy: Allocate to highest severity
- Medium: Handle multiple regions
- Hard: Optimize allocation

## Run
uvicorn server.app:app --reload