import json
import pytest
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
import yaml

console = Console()

RETRIEVAL_CONFIGS = {
    "faiss_only": {
        "ensemble_method": "linear",
        "ranker_weights": {"faiss": 1.0, "bm25": 0.0},
        "top_k": 5,
    },
    "bm25_only": {
        "ensemble_method": "linear",
        "ranker_weights": {"faiss": 0.0, "bm25": 1.0},
        "top_k": 5,
    },
    "hybrid_rrf": {
        "ensemble_method": "rrf",
        "ranker_weights": {"faiss": 0.5, "bm25": 0.5},
        "top_k": 5,
    },
}

INDEX_PREFIXES = {
    "baseline": "baseline_ch3",
    "doc2query": "d2q_ch3",
}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_doc2query_comparison(config, results_dir):
    benchmark_path = Path("tests/ch3.yaml")
    with open(benchmark_path, "r") as f:
        data = yaml.safe_load(f)
        ch3_benchmarks = data.get("benchmarks", [])

    print_test_config(config)

    # Collect all results: results[retrieval_config][index_name] = metrics
    all_results = {}

    for retrieval_name, retrieval_cfg in RETRIEVAL_CONFIGS.items():
        all_results[retrieval_name] = {}
        console.print(f"\n[bold yellow]── Retrieval config: {retrieval_name} ──[/bold yellow]")

        for index_name, index_prefix in INDEX_PREFIXES.items():
            console.print(f"\n[bold cyan]  Evaluating {index_name} ({index_prefix})...[/bold cyan]")
            metrics = evaluate_retrieval(
                ch3_benchmarks,
                config,
                index_prefix,
                results_dir,
                retrieval_cfg,
            )
            if metrics is None:
                console.print(f"[red]  ❌ Skipping {index_name}/{retrieval_name} — artifacts not found.[/red]")
            all_results[retrieval_name][index_name] = metrics

    print_full_summary(all_results, ch3_benchmarks)

    # Save raw results for your writeup
    out = results_dir / "doc2query_full_results.json"
    with open(out, "w") as f:
        # metrics contain floats so default=str handles any non-serializable edge cases
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[dim]Full results saved to {out}[/dim]")


def print_test_config(config):
    print(f"\n{'='*60}")
    print("  Doc2Query Retrieval Comparison — 3-Config Sweep")
    print(f"{'='*60}")
    print(f"  Embedding Model: {Path(config['embed_model']).name if '/' in config['embed_model'] else config['embed_model']}")
    print(f"  Indexes:         {', '.join(INDEX_PREFIXES.keys())}")
    print(f"  Retrieval cfgs:  {', '.join(RETRIEVAL_CONFIGS.keys())}")
    print(f"  Top-K:           5")
    print(f"{'='*60}\n")


def evaluate_retrieval(benchmarks, config, index_prefix, results_dir, retrieval_cfg):
    from src.config import RAGConfig
    from src.retriever import load_artifacts, FAISSRetriever, BM25Retriever
    from src.ranking.ranker import EnsembleRanker

    cfg = RAGConfig(
        embed_model=config.get("embed_model"),
        ensemble_method=retrieval_cfg["ensemble_method"],
        ranker_weights=retrieval_cfg["ranker_weights"],
        top_k=retrieval_cfg["top_k"],
        rrf_k=60,
    )

    artifacts_dir = cfg.get_artifacts_directory(partial=True)

    try:
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, index_prefix)
    except FileNotFoundError:
        console.print(f"    [red]Artifacts for '{index_prefix}' not found in {artifacts_dir}.[/red]")
        return None

    retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    total_recall = 0.0
    total_latency = 0.0
    details = []

    for item in benchmarks:
        benchmark_id = item.get("id", "unknown")
        query = item["question"]
        keywords = item.get("keywords", [])

        t0 = time.perf_counter()

        raw_scores = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(query, cfg.num_candidates, chunks)

        ordered, _ = ranker.rank(raw_scores=raw_scores)
        topk_idxs = ordered[:cfg.top_k]
        retrieved_text = " ".join([chunks[i].lower() for i in topk_idxs])

        latency = time.perf_counter() - t0
        total_latency += latency

        recall = 0.0
        if keywords:
            hits = sum(1 for kw in keywords if kw.lower() in retrieved_text)
            recall = hits / len(keywords)

        total_recall += recall
        details.append({"id": benchmark_id, "recall": recall, "latency": latency})
        print(f"    • {benchmark_id:25} | Recall: {recall:.1%} | {latency:.4f}s")

    n = len(benchmarks)
    return {
        "avg_recall": total_recall / n if n else 0,
        "avg_latency": total_latency / n if n else 0,
        "details": details,
    }


def print_full_summary(all_results, benchmarks):
    # ── Per-config comparison table (baseline vs doc2query) ──
    for retrieval_name, index_results in all_results.items():
        base = index_results.get("baseline")
        d2q  = index_results.get("doc2query")

        if not base or not d2q:
            continue

        table = Table(title=f"[{retrieval_name}]  Baseline vs Doc2Query")
        table.add_column("Metric",     justify="left",  style="cyan", no_wrap=True)
        table.add_column("Baseline",   justify="right", style="red")
        table.add_column("Doc2Query",  justify="right", style="green")
        table.add_column("Delta",      justify="right", style="yellow")

        recall_diff  = (d2q["avg_recall"]  - base["avg_recall"])  * 100
        latency_diff =  d2q["avg_latency"] - base["avg_latency"]

        table.add_row(
            "Avg Keyword Recall",
            f"{base['avg_recall']:.1%}",
            f"{d2q['avg_recall']:.1%}",
            f"{recall_diff:+.1f}%",
        )
        table.add_row(
            "Avg Query Latency",
            f"{base['avg_latency']:.4f}s",
            f"{d2q['avg_latency']:.4f}s",
            f"{latency_diff:+.4f}s",
        )
        console.print("\n")
        console.print(table)

        # Per-question breakdown
        console.print(f"\n[bold]Question breakdown — {retrieval_name}:[/bold]")
        for i, bm in enumerate(benchmarks):
            qid      = bm["id"]
            base_rec = base["details"][i]["recall"]
            d2q_rec  = d2q["details"][i]["recall"]
            arrow    = "↑" if d2q_rec > base_rec else ("↓" if d2q_rec < base_rec else "=")
            color    = "green" if d2q_rec > base_rec else ("red" if d2q_rec < base_rec else "white")
            console.print(
                f"  - {qid:26}: Baseline {base_rec:>5.1%} | D2Q {d2q_rec:>5.1%} [{color}]{arrow}[/{color}]"
            )

    # ── Cross-config summary (one row per retrieval config) ──
    console.print("\n")
    summary = Table(title="Cross-Config Summary (Doc2Query delta over Baseline)")
    summary.add_column("Retrieval Config", style="cyan")
    summary.add_column("Baseline Recall",  justify="right")
    summary.add_column("D2Q Recall",       justify="right")
    summary.add_column("Delta",            justify="right", style="yellow")

    for retrieval_name, index_results in all_results.items():
        base = index_results.get("baseline")
        d2q  = index_results.get("doc2query")
        if not base or not d2q:
            summary.add_row(retrieval_name, "N/A", "N/A", "N/A")
            continue
        delta = (d2q["avg_recall"] - base["avg_recall"]) * 100
        summary.add_row(
            retrieval_name,
            f"{base['avg_recall']:.1%}",
            f"{d2q['avg_recall']:.1%}",
            f"{delta:+.1f}%",
        )

    console.print(summary)
    print(f"\n{'='*60}\n")