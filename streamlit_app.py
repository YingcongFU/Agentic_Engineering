import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI

client = OpenAI()

project_directory = Path(__file__).resolve().parent
storage_directory = project_directory / "vector_store"

index = faiss.read_index(
    str(storage_directory / "novatech.index")
)

with open(
    storage_directory / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    chunk_metadata = json.load(file)