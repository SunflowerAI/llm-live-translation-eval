from typedefinitions import *
from openrouter_inference_source import *
from dataset import SENTENCES_LIST
from utils import md5hash, get_translation_with_cache_check
from secrets_env import *
import time
import concurrent.futures

from aquarel import load_theme
import matplotlib.pyplot as plt

import re


def process_single_task(
    model, sentence_cat, sentence_to_test, language, evaluators, depth, cache
):
    results = []
    last_iteration_english = sentence_to_test

    for n in range(1, depth + 1):
        print("Depth", n)

        iteration_target = get_translation_with_cache_check(
            model,
            TranslatableLanguage.English,
            language,
            "Coherence-" + sentence_cat,
            last_iteration_english,
            cache,
        )
        print("Target translated", iteration_target)

        last_iteration_english = get_translation_with_cache_check(
            model,
            language,
            TranslatableLanguage.English,
            "Coherence-" + sentence_cat,
            iteration_target,
            cache,
        )
        print("English backtranslated", last_iteration_english)

        # (str, int)
        evaluations_out = []

        for evaluator_str, evaluator_inference in evaluators:
            prompt = f"""Your role is to rate how close two English sentences in a game of chinese whispers are, from 0 to 100. Evaluate based on how close it is to the original *meaning*, tone, etc.
Original text:
```{sentence_to_test}```
New text:
```{last_iteration_english}```

You may think aloud in truly minimal, terse points for your own use, saving tokens. BE ULTRAPRECISE!
Your final response should be a single integer between 0 and 100, on a NEW LINE, with NOTHING ELSE on that LAST LINE!"""

            score = None
            while True:
                eval_temp = 0
                cache_key = md5hash(prompt) + "|" + evaluator_str + "|" + str(eval_temp)
                print(cache_key)

                resp = None
                if cache.get(cache_key):
                    resp = cache.get(cache_key)
                else:
                    try:
                        resp = evaluator_inference.infer(prompt, eval_temp)
                        cache.set(cache_key, resp)
                    except Exception as e:
                        print("Err on inf for", evaluator_str, ":", e)
                        continue

                try:
                    score = int(re.findall(r"\d{1,3}", resp[-10:])[-1])
                    if score >= 0 and score <= 100:
                        print("Got score", score)
                        break
                    else:
                        print("Failed to parse correct score from", resp)
                except Exception as e:
                    print("Err on parsing resp", resp)

                if eval_temp < 1:
                    eval_temp += 0.1

                time.sleep(4)

            evaluations_out.append((evaluator_str, score))

        results.append(
            CoherenceIteration(
                sentence_cat=sentence_cat,
                sentence=sentence_to_test,
                depth=n,
                staged_text=last_iteration_english,
                evaluations=evaluations_out,
                tested_entry=model,
            )
        )

    return results


def inference_coherence_batch(
    language, to_test, evaluators, depth, first_x_in_cat, temp, cache, max_workers=None
):
    sentences_list = []
    for cat, sentences in SENTENCES_LIST.items():
        for x in sentences[:first_x_in_cat]:
            sentences_list.append((cat, x))

    out_data = []
    tasks = []

    # Create all tasks
    for model in to_test:
        for sentence_cat, sentence_to_test in sentences_list:
            tasks.append((model, sentence_cat, sentence_to_test))

    # Process tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map tasks to futures
        futures = [
            executor.submit(
                process_single_task,
                model,
                sentence_cat,
                sentence_to_test,
                language,
                evaluators,
                depth,
                cache,
            )
            for model, sentence_cat, sentence_to_test in tasks
        ]

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                out_data.extend(result)
            except Exception as e:
                print(f"Error in task: {e}")

    return out_data


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Optional, Set, Dict, Tuple


def chart_coherence(
    dataset,
    categories=None,
    title="Coherence vs Translation Depth",
    figsize=(10, 6),
    save_path=None,
):
    """
    Chart coherence scores over translation depth for different models.

    Args:
        dataset: List of CoherenceIteration objects
        categories: Optional list of categories to include. If None, includes all categories.
        title: Title for the chart
        figsize: Figure size as (width, height) tuple
        save_path: Optional path to save the figure. If None, the figure is displayed but not saved.

    Returns:
        matplotlib Figure object
    """
    # Extract unique models and find maximum depth
    models_being_tested = []
    max_depth = 0

    for item in dataset:
        if categories is None or item.sentence_cat in categories:
            if item.tested_entry not in models_being_tested:
                models_being_tested.append(item.tested_entry)

            if item.depth > max_depth:
                max_depth = item.depth

    models_being_tested = list(models_being_tested)

    theme = load_theme("scientific")
    theme.apply()

    # Organize data by model and depth
    # Structure: {model: {depth: [scores]}}
    model_depth_scores = defaultdict(lambda: defaultdict(list))

    for item in dataset:
        print(item)
        if categories is None or item.sentence_cat in categories:
            # Average all evaluation scores for this iteration
            avg_score = np.mean([score for _, score in item.evaluations])
            model_depth_scores[item.tested_entry.unique_id()][item.depth].append(
                avg_score
            )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot perfect coherence baseline at 100
    x_range = range(0, max_depth + 1)
    ax.plot(
        x_range,
        [100] * len(x_range),
        "--",
        color="gray",
        alpha=0.5,
        label="Perfect Coherence",
    )

    # Plot data for each model
    for model in models_being_tested:
        # Compute average scores at each depth
        colour = model.model_company.colour()
        depths = [0]
        avg_scores = [100]

        for depth in range(1, max_depth + 1):
            if depth in model_depth_scores[model.unique_id()]:
                depths.append(depth)
                avg_scores.append(np.mean(model_depth_scores[model.unique_id()][depth]))

        # Plot line for this model
        ax.plot(
            depths,
            avg_scores,
            "o-",
            label=f"{model.model_name.value} ({model.model_company.value})",
            color=colour,
        )

    # Set up axes and labels
    ax.set_xlabel("Translation Depth")
    ax.set_ylabel("Average Coherence Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show integer values only
    ax.set_xticks(range(0, max_depth + 1))

    # Add legend
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

    # Ensure y-axis starts at 0 and goes to 100 or slightly above
    # ax.set_ylim(0, 105)

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=600)

    return fig


def chart_coherence_by_category(
    dataset,
    models=None,
    title="Coherence by Category",
    figsize=(12, 8),
    depth=1,
    save_path=None,
):
    """
    Chart coherence scores for each category at a specific depth.

    Args:
        dataset: List of CoherenceIteration objects
        models: Optional list of models to include. If None, includes all models.
        title: Title for the chart
        figsize: Figure size as (width, height) tuple
        depth: Translation depth to analyze
        save_path: Optional path to save the figure. If None, the figure is displayed but not saved.

    Returns:
        matplotlib Figure object
    """
    # Extract unique models and categories
    all_models = []
    all_categories = set()

    for item in dataset:
        if item.depth == depth:
            all_models.append(item.tested_entry)
            all_categories.add(item.sentence_cat)

    # Filter models if specified
    if models is not None:
        models_to_use = [m for m in all_models if m in models]
    else:
        models_to_use = list(all_models)

    all_categories = list(all_categories)

    theme = load_theme("scientific")
    theme.apply()

    # Organize data by model and category
    # Structure: {model: {category: [scores]}}
    model_cat_scores = defaultdict(lambda: defaultdict(list))

    for item in dataset:
        if item.depth == depth and item.tested_entry in models_to_use:
            # Average all evaluation scores for this iteration
            avg_score = np.mean([score for _, score in item.evaluations])
            model_cat_scores[item.tested_entry][item.sentence_cat].append(avg_score)

    # Compute average scores for each model and category
    model_cat_avg = {model: {} for model in models_to_use}
    for model in models_to_use:
        for cat in all_categories:
            if cat in model_cat_scores[model]:
                model_cat_avg[model][cat] = np.mean(model_cat_scores[model][cat])
            else:
                model_cat_avg[model][cat] = 0  # No data for this combination

    # Create figure for bar chart
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    n_models = len(models_to_use)
    n_cats = len(all_categories)
    width = 0.8 / n_models

    # Plot bars for each model and category
    for i, model in enumerate(models_to_use):
        positions = np.arange(n_cats) + (i - n_models / 2 + 0.5) * width
        values = [model_cat_avg[model].get(cat, 0) for cat in all_categories]
        ax.bar(positions, values, width, label=str(model))

    # Set up axes and labels
    ax.set_xlabel("Sentence Category")
    ax.set_ylabel("Average Coherence Score")
    ax.set_title(f"{title} (Depth {depth})")
    ax.set_xticks(np.arange(n_cats))
    ax.set_xticklabels(all_categories, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend
    ax.legend()

    # Ensure y-axis goes to 100 or slightly above
    ax.set_ylim(0, 105)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig
