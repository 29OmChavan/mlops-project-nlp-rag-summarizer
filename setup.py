import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "1.0.0"

REPO_NAME = "mlops-project-3-nlp-rag-summarizer"
AUTHOR_USER_NAME = "29OmChavan"
SRC_REPO = "nlp_rag_system"
AUTHOR_EMAIL = "29omchavan@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Production NLP & LLMOps System: Transformer LoRA Fine-Tuning, Advanced Hybrid RAG with Citations, ROUGE/BLEU Gate, and LLM Monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
