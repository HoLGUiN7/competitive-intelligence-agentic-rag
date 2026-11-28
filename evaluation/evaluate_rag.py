import json
import os
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError(
        "Falta la variable GROQ_API_KEY. Ejecuta en PowerShell: $env:GROQ_API_KEY='tu_clave'"
    )

dataset_path = Path(__file__).parent / "eval_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

eval_dataset = Dataset.from_list(raw_data)

# Usamos openai/gpt-oss-120b en Groq como juez y max_workers=1 para evitar rate limits (429)
judge_llm = LangchainLLMWrapper(
    ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=8192)
)
run_config = RunConfig(max_workers=1, max_retries=3, timeout=180)

print("Ejecutando evaluación de Faithfulness con Ragas usando Groq (openai/gpt-oss-120b)...\n")
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        Faithfulness(),
    ],
    llm=judge_llm,
    run_config=run_config,
)

df = results.to_pandas()
print("=== RESULTADOS POR CASO ===")
print(df[["user_input", "faithfulness"]].to_string(index=False))

avg_val = None if df["faithfulness"].isna().all() else round(float(df["faithfulness"].mean()), 4)
summary = {
    "judge_provider": "groq",
    "judge_model": "openai/gpt-oss-120b",
    "faithfulness_avg": avg_val,
    "details": df.to_dict(orient="records"),
}

output_path = Path(__file__).parent / "ragas_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n=== PROMEDIO GLOBAL ===")
print(f"Faithfulness (Fidelidad al contexto): {f'{avg_val * 100:.2f}%' if avg_val is not None else 'N/A (NaN)'}")
print(f"\nReporte guardado en: {output_path}")