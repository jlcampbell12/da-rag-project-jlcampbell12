
# Welcome
This repo is an exercise to create a **RAG architecture using CapTech-provided models to answer questions about specific topics**. This is a hands-on follow up to the training in the [D&A AI Technical Training Path 1](https://captechventuresinc.sharepoint.com/sites/CapTechCollege/SitePages/DAAITechnicalTrainingPath1.aspx). For additional background and relevant courses, see [D&A AI Technical Foundations Playlist](https://captechventuresinc.sharepoint.com/sites/CapTechCollege/SitePages/DAAITechnicalFoundationsPlaylist.aspx) and [D&A AI Technical Training Path 2](https://captechventuresinc.sharepoint.com/sites/CapTechCollege/SitePages/DAAITechnicalTrainingPath2.aspx).

This starter-pack attempts to give you sufficient guardrails and constraints such that you can implement a complete RAG-based system with about a half day of effort using a conversational coding assistant such as GitHub Copilot. Some examples of these constraints:
* the content of the RAG dataset is provided through [Hugging Face](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) as a curated, pre-chunked set of wikipedia articles, along with 918 pre-labeled questions and answers for that data
* the dozens of choices around an automation engine, a vector database, and other architecture has been made -- we'll use the [LlamaIndex](https://github.com/run-llama/llama_index) tooling, including the LlamaIndex-provided `VectorStoreIndex` as a local vector database
* we'll use the CapTech-provided [Direct Model Access](https://captechventuresinc.sharepoint.com/sites/AIHub/SitePages/direct-model-access.aspx) with given python code so you don't get bogged down with authentication issues: you can focus on the RAG architecture instead

## What's in this repo
In this repo you'll see a recommended folder structure, several markdown files in docs/ and several python example files for using the CapTech-provided models via LlamaIndex. There is an `instructions.md` file, a README.md, and STATUS.md that you can use to control how your coding assistant approaches the task of creating a **RAG-backed question / answer system**. 

You can have your coding assistant update the uppercase files (README, STATUS) with current information about the system as it evolves. Try to keep this file (readme_setup.md) in its original state. Feel free to add your own instructions to `instructions.md` or replace it completely. The given contents of `instructions.md` are some instructions that I have found useful in keeping my coding assistant (usually Claude Sonnet 4 running in agent-mode in GitHub Copilot in VS Code) on track with this particular type of exercise.

## Evalutation
You should create several Jupyter notebooks that demonstrate various aspects of the **RAG-backed question / answer system** that you create. Demonstrate feeding in some of the questions from the Hugging Face dataset and get the expected answers. Demonstrate some of the inner workings of your system: how many documents are loaded into the vector database; what an embedding looks like; what your end-to-end question / answer process looks like; etc. 

Your project should be sufficiently mature that you should be able to do a live demo of your running project. For this exercise you aren't required to do that kind of demo, although you are definitely encouraged to demo to your coach, your coachees, and the (DE)monstrations audience. For this exercise, your git commits and files will be analyzed by automated tooling to ensure that your project repo has grown from the starter-pack state to a state that shows you added code, documentation, tests, and notebooks. 


# Setup

## Prerequisites
We are using [uv](https://github.com/astral-sh/uv) for both python environment management and dependency management. There are many options in this space (venv, pipenv, poetry, pyenv). `uv` has been tested with CapTech-issued Macs and PCs. The official `uv` docs have alternate instructions for installation. The instructions below have been tested with CapTech-issued laptops. 

The following should already be available or installed (fill out IT requests if necessary)
* git tooling
* a GitHub account with access to the @captechconsulting organization
* GitHub Copilot
* a code editor (VS Code or otherwise) that is set up to use GitHub Copilot through the @captechconsulting organization (Claude Code or any other assistant is also fine)

### Windows-only
* download the azure-dev-cli (aka "azd") from https://github.com/Azure/azure-dev/releases
* download uv from https://github.com/astral-sh/uv/releases
* unzip both and copy someplace on the path -- if you don't already have someplace, then create a directory at c:\utils and put that on the system path
* if the azd binary is named something like "azd-windows-amd64.exe" rename it to "azd.exe" to follow the instructions below

### Mac-only
* `brew tap azure/azd && brew install azd`
* `brew install uv`

## Github setup
* at https://github.com/captechconsulting/da-rag-project-2026-starter-pack use the `Code` button to "Download ZIP"
* in your own github account create a new repo called something like `da-rag-project-2026-<your-github-username>`
* clone the new repo from your account to your machine
* copy the files from the downloaded zip file into the directory for you cloned repo
* add, commit, and push into your repo

## Environment Setup
* in your project directory, set up a new virtual environment - `uv venv --python 3.13`
* use the given pyproject.toml to pull python libraries - `uv sync`
* add new python libraries like this: `uv add jupyterlab`
* run installed utilities like this: `uv run pytest tests/`
* run scripts with the correct environment like this: `uv run python my_script.py`

## VS Code Setup
After you create and populate the virtual environment, you should be able to bring up the VS Code command palette (cmd-shift-p) and choose "Select Python Interpreter". The "recommended" option should be the path you just created (.venv/bin/python in the project folder). Selecting the interpreter that is used by the virtual environment will help linters (like Pylance) resolve imports correctly. 

I use a pretty fancy command line as part of my zsh setup on the Mac (colors, formatted time and repo information, etc.). This kind of messes up the agent-mode interaction. So, I spend a few minutes at the beginning of a project getting the agent to create a default terminal configuration for the project (will be in .vscode/settings.json) with a bash terminal with a very plain prompt as the default. 

# Ready to go
Have your agent analyze the files in the repo (instructions.md, README.md, STATUS.md, docs/) to build a context for the project. Test early and often. 

Give feedback on this exercise and ask questions at https://github.com/captechconsulting/da-rag-project-2026-starter-pack


 

-- Tim Chase, November, 2025
