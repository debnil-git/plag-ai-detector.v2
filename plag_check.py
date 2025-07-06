# ------------------------------
# File: plag_check.py (v10.4) - FIXED
# Enhanced with multiple graph types and better visualization
# ------------------------------

import aiohttp
import asyncio
import random
import re
import torch
import docx2txt
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from collections import Counter
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# --- Load Models (These will be loaded when imported) ---
# Models are loaded lazily to improve startup time
import warnings
warnings.filterwarnings("ignore")  # Suppress transformer warnings

try:
    print("📥 Loading sentence transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("📥 Loading GPT-2 model...")
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# --- Extraction ---
def extract_text_from_docx(path):
    return docx2txt.process(path)

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# --- Async Search ---
async def fetch_html(session, url):
    try:
        async with session.get(url, timeout=10) as resp:
            return await resp.text()
    except:
        return ""

async def fetch_web_text(query, engine="ddg", num_results=10):
    await asyncio.sleep(random.uniform(1.5, 3.5))  # mimic human behavior
    headers = {"User-Agent": "Mozilla/5.0"}
    results = []

    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            if engine == "ddg":
                search_url = f"https://html.duckduckgo.com/html/?q={query}"
                html = await fetch_html(session, search_url)
                soup = BeautifulSoup(html, "html.parser")
                links = soup.find_all("a", class_="result__a", limit=num_results)
                results = [a["href"] for a in links]
            elif engine == "mojeek":
                search_url = f"https://www.mojeek.com/search?q={query}"
                html = await fetch_html(session, search_url)
                soup = BeautifulSoup(html, "html.parser")
                links = soup.find_all("a", href=True)
                results = [a["href"] for a in links if "mojeek" not in a["href"]][:num_results]

        except:
            return ""

        texts = []
        for url in results:
            try:
                html = await fetch_html(session, url)
                soup = BeautifulSoup(html, "html.parser")
                body = soup.get_text(separator=" ", strip=True)
                texts.append(body[:500])
            except:
                continue

        return " ".join(texts)

# --- Similarity Check ---
async def check_similarity(text):
    try:
        web_text = await fetch_web_text(text, engine="ddg")
    except:
        web_text = await fetch_web_text(text, engine="mojeek")

    if not web_text:
        return 0.0, "No match found", "N/A"

    emb1 = model.encode(text, convert_to_tensor=True)
    emb2 = model.encode(web_text, convert_to_tensor=True)
    sim = round(float(util.cos_sim(emb1, emb2).item()) * 100, 2)

    verdict = "Copied" if sim > 75 else "Unique"
    return sim, verdict, "duckduckgo.com"

# --- AI Detection ---
def detect_ai(text):
    input_ids = gpt_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        loss = gpt_model(input_ids, labels=input_ids).loss
    perplexity = torch.exp(loss).item()
    verdict = "Likely AI" if perplexity < 60 else "Human-like"
    return verdict, round(perplexity, 2)

# --- Combined Async Analyzer ---
async def analyze_sentences_async(sentences):
    results = []
    for sentence in sentences:
        sim_score, verdict, source = await check_similarity(sentence)
        ai_verdict, ai_score = detect_ai(sentence)
        results.append((sentence, sim_score, verdict, source, "ai", ai_verdict, ai_score))
    return results

# --- AI-Only Async Analyzer ---
async def analyze_ai_only(sentences):
    results = []
    for sentence in sentences:
        verdict, score = detect_ai(sentence)
        results.append((sentence, verdict, score))
    return results

# --- Enhanced Graph Generation ---
def generate_graphs(ai_results, plag_results=None, sources=None, mode="full", graph1_type="pie_chart", graph2_type="ai_trend"):
    """
    Generate multiple types of graphs based on analysis results
    """
    graph_paths = {}
    
    try:
        # Set style for better looking graphs
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        # Create graphs directory
        os.makedirs("graphs", exist_ok=True)
        
        # Generate first graph
        graph1_path = generate_single_graph(ai_results, plag_results, sources, graph1_type, mode, "graph1")
        if graph1_path:
            graph_paths["graph1"] = graph1_path
        
        # Generate second graph
        graph2_path = generate_single_graph(ai_results, plag_results, sources, graph2_type, mode, "graph2")
        if graph2_path:
            graph_paths["graph2"] = graph2_path
            
    except Exception as e:
        print(f"Graph generation error: {e}")
        return None
    
    return graph_paths if graph_paths else None

def generate_single_graph(ai_results, plag_results, sources, graph_type, mode, filename):
    """Generate a single graph based on type"""
    
    try:
        if graph_type == "pie_chart":
            return generate_pie_chart(ai_results, plag_results, mode, filename)
        elif graph_type == "ai_trend":
            return generate_ai_trend(ai_results, filename)
        elif graph_type == "plagiarism_trend":
            return generate_plagiarism_trend(plag_results, filename)
        elif graph_type == "source_freq":
            return generate_source_frequency(sources, filename)
        elif graph_type == "stacked_bar":
            return generate_stacked_bar(ai_results, plag_results, mode, filename)
        elif graph_type == "radar_chart":
            return generate_radar_chart(ai_results, plag_results, mode, filename)
        else:
            return None
    except Exception as e:
        print(f"Error generating {graph_type}: {e}")
        return None

def generate_pie_chart(ai_results, plag_results, mode, filename):
    """Generate pie chart showing content breakdown"""
    plt.figure(figsize=(10, 8))
    
    if mode == "full" and plag_results:
        # Full analysis mode
        ai_count = sum(1 for _, verdict, _ in ai_results if "AI" in verdict)
        plag_count = sum(1 for _, verdict, _ in plag_results if "Copied" in verdict)
        clean_count = len(ai_results) - ai_count - plag_count
        
        sizes = [ai_count, plag_count, clean_count]
        labels = ['AI Generated', 'Plagiarized', 'Original']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        explode = (0.1, 0.1, 0)
        
    else:
        # AI-only mode
        ai_count = sum(1 for _, verdict, _ in ai_results if "AI" in verdict)
        human_count = len(ai_results) - ai_count
        
        sizes = [ai_count, human_count]
        labels = ['AI Generated', 'Human-like']
        colors = ['#ff6b6b', '#45b7d1']
        explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Content Analysis Breakdown', fontsize=16, fontweight='bold')
    plt.axis('equal')
    
    path = f"graphs/{filename}_pie.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def generate_ai_trend(ai_results, filename):
    """Generate AI detection trend line"""
    plt.figure(figsize=(12, 6))
    
    # Extract AI scores
    scores = [float(score) for _, _, score in ai_results]
    sentence_nums = list(range(1, len(scores) + 1))
    
    plt.plot(sentence_nums, scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=60, color='r', linestyle='--', label='AI Threshold (60)')
    plt.fill_between(sentence_nums, scores, alpha=0.3)
    
    plt.xlabel('Sentence Number')
    plt.ylabel('Perplexity Score')
    plt.title('AI Detection Trend (Lower = More AI-like)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = f"graphs/{filename}_ai_trend.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def generate_plagiarism_trend(plag_results, filename):
    """Generate plagiarism detection trend - FIXED"""
    if not plag_results:
        return None
        
    plt.figure(figsize=(12, 6))
    
    # FIXED: Extract similarity scores correctly (index 1 is the score, index 2 is the verdict)
    scores = [float(score) for _, score, _ in plag_results]  # score is at index 1
    sentence_nums = list(range(1, len(scores) + 1))
    
    plt.plot(sentence_nums, scores, 'g-', linewidth=2, marker='s', markersize=4)
    plt.axhline(y=75, color='r', linestyle='--', label='Plagiarism Threshold (75%)')
    plt.fill_between(sentence_nums, scores, alpha=0.3)
    
    plt.xlabel('Sentence Number')
    plt.ylabel('Similarity Score (%)')
    plt.title('Plagiarism Detection Trend', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = f"graphs/{filename}_plag_trend.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def generate_source_frequency(sources, filename):
    """Generate source frequency distribution"""
    if not sources:
        return None
        
    plt.figure(figsize=(10, 6))
    
    # Count sources
    source_counts = Counter(sources)
    sources_list = list(source_counts.keys())[:10]  # Top 10
    counts = [source_counts[s] for s in sources_list]
    
    plt.bar(range(len(sources_list)), counts, color='skyblue')
    plt.xlabel('Sources')
    plt.ylabel('Frequency')
    plt.title('Top Sources Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(sources_list)), sources_list, rotation=45, ha='right')
    
    path = f"graphs/{filename}_sources.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def generate_stacked_bar(ai_results, plag_results, mode, filename):
    """Generate stacked bar chart by sections - FIXED"""
    plt.figure(figsize=(10, 6))
    
    # Split into sections
    section_size = max(1, len(ai_results) // 5)  # 5 sections
    sections = []
    
    for i in range(0, len(ai_results), section_size):
        section = ai_results[i:i+section_size]
        ai_count = sum(1 for _, verdict, _ in section if "AI" in verdict)
        
        if mode == "full" and plag_results:
            plag_section = plag_results[i:i+section_size]
            # FIXED: Check verdict correctly (index 2 is the verdict)
            plag_count = sum(1 for _, _, verdict in plag_section if "Copied" in verdict)
            clean_count = len(section) - ai_count - plag_count
            sections.append([ai_count, plag_count, clean_count])
        else:
            human_count = len(section) - ai_count
            sections.append([ai_count, human_count])
    
    # Create stacked bar chart
    section_names = [f'Section {i+1}' for i in range(len(sections))]
    
    if mode == "full" and plag_results:
        ai_counts = [s[0] for s in sections]
        plag_counts = [s[1] for s in sections]
        clean_counts = [s[2] for s in sections]
        
        plt.bar(section_names, ai_counts, label='AI Generated', color='#ff6b6b')
        plt.bar(section_names, plag_counts, bottom=ai_counts, label='Plagiarized', color='#4ecdc4')
        plt.bar(section_names, clean_counts, bottom=[a+p for a,p in zip(ai_counts, plag_counts)], 
                label='Original', color='#45b7d1')
    else:
        ai_counts = [s[0] for s in sections]
        human_counts = [s[1] for s in sections]
        
        plt.bar(section_names, ai_counts, label='AI Generated', color='#ff6b6b')
        plt.bar(section_names, human_counts, bottom=ai_counts, label='Human-like', color='#45b7d1')
    
    plt.xlabel('Document Sections')
    plt.ylabel('Number of Sentences')
    plt.title('Content Analysis by Section', fontsize=14, fontweight='bold')
    plt.legend()
    plt.xticks(rotation=45)
    
    path = f"graphs/{filename}_stacked.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

def generate_radar_chart(ai_results, plag_results, mode, filename):
    """Generate radar chart showing multiple metrics - FIXED"""
    plt.figure(figsize=(8, 8))
    
    # Calculate metrics
    total_sentences = len(ai_results)
    ai_percentage = sum(1 for _, verdict, _ in ai_results if "AI" in verdict) / total_sentences * 100
    
    if mode == "full" and plag_results:
        # FIXED: Check verdict correctly (index 2 is the verdict, index 1 is the score)
        plag_percentage = sum(1 for _, _, verdict in plag_results if "Copied" in verdict) / total_sentences * 100
        avg_similarity = sum(float(score) for _, score, _ in plag_results) / total_sentences
        originality = 100 - max(ai_percentage, plag_percentage)
    else:
        plag_percentage = 0
        avg_similarity = 0
        originality = 100 - ai_percentage
    
    avg_perplexity = sum(float(score) for _, _, score in ai_results) / total_sentences
    
    # Normalize perplexity to 0-100 scale
    perplexity_normalized = min(100, max(0, (avg_perplexity - 20) / 80 * 100))
    
    # Radar chart data
    categories = ['AI Detection', 'Plagiarism', 'Similarity', 'Originality', 'Perplexity']
    values = [ai_percentage, plag_percentage, avg_similarity, originality, perplexity_normalized]
    
    # Add first point to close the circle
    values += values[:1]
    
    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Create radar chart
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, color='#ff6b6b')
    ax.fill(angles, values, alpha=0.25, color='#ff6b6b')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    
    plt.title('Multi-Metric Analysis', size=14, fontweight='bold', pad=20)
    
    path = f"graphs/{filename}_radar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path