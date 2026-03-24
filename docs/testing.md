# AI Lab Model Testing Guide

This document explains the testing system for the AI Lab authentication and model access system.

## 📊 Quick Overview

- **XX total tests** across X test files
- **Default**: `uv run pytest` 

## 🏗️ Test Organization

Tests run in fail-fast order with numbered prefixes:

- `test_00_*` - **Smoke Tests**: 
- `test_10_*` - **Auth Tests**: 
- `test_20_*` - **Basic Tests**: 
- `test_30_*` - **Model Tests**: 
- `test_50_*` - **Integration Tests**: 

## ⚡ Test Tiers



## 🧪 What Gets Tested



## 🚀 Common Commands



## 🔧 When Tests Fail

Tests provide specific guidance based on failure type:

**Environment failures** → Check Azure auth: `azd auth login --scope api://ailab/Model.Access`

**Authentication failures** → Refresh login, verify permissions

**Model access failures** → Check MODEL_REGISTRY, verify deployments

**Integration failures** → Check quotas, API versions, network connectivity

## ✅ What's Covered

- ✅ **Azure authentication** with controlled token providers
- ✅ **Model isolation** preventing unauthorized access  
- ✅ **LlamaIndex integration** with chat, embeddings, vector stores
- ✅ **Real API calls** validating end-to-end functionality
- ✅ **Example patterns** ensuring documentation accuracy
- ✅ **Security controls** blocking unauthorized models
- ✅ **Developer experience** with clear error messages
- ✅ **Clean output** with dependency warnings filtered

The default `uv run pytest` gives fast feedback for development, while full tests ensure comprehensive validation before releases.