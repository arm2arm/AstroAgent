CrewAI Astronomy Research Workflow System
Practical Multi-Agent System for Astronomical Data Analysis
Working Implementation Guide
February 2026

Executive Summary
This document provides a complete, working implementation of a CrewAI-based multi-agent system for astronomy research workflows. The system uses a local LLM endpoint (https://ai.aip.de/api), features a production-ready Streamlit interface, and focuses on practical Gaia DR3 data analysis tasks.
Key Features:
Simple Architecture: 4 core agents (Planner, Analyst, Coder, Reviewer)
Local LLM: OpenAI-compatible endpoint at AIP
Real Workflows: Gaia DR3 stellar analysis, HR diagrams, kinematics
Production UI: Clean Streamlit dashboard with monitoring
Fully Working: Copy-paste ready code with detailed setup

Table of Contents
System Architecture
Complete Code Implementation
Streamlit Dashboard
Installation and Setup
Usage Examples
Configuration Guide

1. System Architecture
Simplified Design
┌─────────────────────────────────────────────────┐
│ Streamlit Web Interface │
│ (Create Workflows, Monitor, View Results) │
└──────────────────┬──────────────────────────────┘
│
┌──────────────────▼──────────────────────────────┐
│ Workflow Orchestrator │
│ (CrewAI Flow with 4 Agents) │
│ │
│ Planner → Analyst → Coder → Reviewer │
└──────────────────┬──────────────────────────────┘
│
┌──────────────────▼──────────────────────────────┐
│ Local LLM Endpoint │
│ https://ai.aip.de/api  │
│ (OpenAI-compatible API) │
└──────────────────────────────────────────────────┘
Agent Roles
Agent
Purpose
Output
Planner
Design analysis strategy
Analysis plan with steps
Analyst
Statistical analysis design
Methods and approach
Coder
Generate Python code
Complete analysis script
Reviewer
Code validation
Quality report

Complete Code Implementation
2.1 Project Structure
astronomy-crewai/
├── app.py  # Streamlit dashboard
├── workflow.py  # CrewAI workflow logic
├── agents.py  # Agent definitions
├── config.py  # Configuration
├── requirements.txt # Dependencies
├── .env # Environment variables
└── outputs/ # Generated workflows
├── workflows/
└── results/


