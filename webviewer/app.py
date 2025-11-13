#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = "/root/et/verl"
RESULTS_DIR = "/root/et/verl/outputs/inference_run/results"
ALLOWED_IMAGE_ROOT = "/root/et"


def _safe_path(p: str) -> str:
    ap = os.path.abspath(p)
    if not ap.startswith(os.path.abspath(ALLOWED_IMAGE_ROOT) + os.sep) and ap != os.path.abspath(ALLOWED_IMAGE_ROOT):
        raise HTTPException(status_code=400, detail="非法路径")
    return ap


def list_jsonl_files() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.isdir(RESULTS_DIR):
        return items
    for name in sorted(os.listdir(RESULTS_DIR)):
        if not name.endswith('.jsonl'):
            continue
        fp = os.path.join(RESULTS_DIR, name)
        try:
            st = os.stat(fp)
            items.append({
                "name": name,
                "path": fp,
                "size": st.st_size,
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            })
        except FileNotFoundError:
            continue
    # 按修改时间降序
    items.sort(key=lambda x: x.get("mtime", ""), reverse=True)
    return items


def iter_jsonl_records(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"_error": "invalid_json", "_raw": line}


def filter_match(rec: Dict[str, Any], query_text: Optional[str]) -> bool:
    if not query_text:
        return True
    q = (query_text or "").lower()
    def s(x: Any) -> str:
        return (str(x) if x is not None else "").lower()
    return any(
        q in s(rec.get(k))
        for k in ["question", "answer", "ground_truth", "model_dir"]
    )


def is_correct(rec: Dict[str, Any]) -> Optional[bool]:
    try:
        reward = rec.get("reward") or {}
        val = reward.get("is_correct")
        if isinstance(val, bool):
            return val
    except Exception:
        pass
    return None


def compute_summary(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(recs)
    correct = sum(1 for r in recs if is_correct(r) is True)
    with_scores = [r for r in recs if isinstance(((r.get("reward") or {}).get("total")), (int, float))]
    avg_total = (sum((r.get("reward") or {}).get("total", 0.0) for r in with_scores) / len(with_scores)) if with_scores else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "avg_total_reward": avg_total,
    }


app = FastAPI(title="Inference Results Viewer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    idx = os.path.join(static_dir, 'index.html')
    if not os.path.exists(idx):
        raise HTTPException(status_code=404, detail="index.html 不存在")
    return FileResponse(idx)


@app.get("/api/files")
def api_files():
    return {"files": list_jsonl_files(), "base_dir": RESULTS_DIR}


@app.get("/api/results")
def api_results(
    file: str = Query(..., description="文件名或绝对路径，若为文件名则从 results 目录解析"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    correctness: str = Query("all", regex="^(all|true|false)$"),
    query: Optional[str] = Query(None),
):
    # 解析文件路径
    fp = file
    if not os.path.isabs(fp):
        fp = os.path.join(RESULTS_DIR, file)
    fp = os.path.abspath(fp)
    if not fp.startswith(os.path.abspath(RESULTS_DIR) + os.sep):
        raise HTTPException(status_code=400, detail="仅允许访问 results 目录内文件")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="文件不存在")

    # 加载并过滤
    filtered: List[Dict[str, Any]] = []
    for rec in iter_jsonl_records(fp):
        if query and not filter_match(rec, query):
            continue
        corr = is_correct(rec)
        if correctness == "true" and corr is not True:
            continue
        if correctness == "false" and corr is not False:
            continue
        filtered.append(rec)

    total = len(filtered)
    page_items = filtered[offset: offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page_items,
        "summary": compute_summary(filtered),
    }


@app.get("/api/image")
def api_image(path: str = Query(..., description="绝对路径（受限于 /root/et 前缀）")):
    ap = _safe_path(path)
    if not os.path.exists(ap):
        raise HTTPException(status_code=404, detail="图片不存在")
    return FileResponse(ap)


@app.get("/api/compare")
def api_compare(
    files: str = Query(..., description="逗号分隔的文件名列表"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    correctness: str = Query("all", regex="^(all|true|false|mixed)$"),
    query: Optional[str] = Query(None),
):
    """多模型对比接口：按问题聚合不同模型的回答"""
    file_list = [f.strip() for f in files.split(",") if f.strip()]
    if not file_list:
        raise HTTPException(status_code=400, detail="至少需要选择一个文件")
    
    # 加载所有文件的数据，并建立索引
    # file_data = {
    #   fname: {
    #       "records": [...],
    #       "by_index": { index: rec },
    #       "by_qimg": { (question, tuple(images)): [rec, ...] }
    #   }
    # }
    file_data: Dict[str, Dict[str, Any]] = {}
    for fname in file_list:
        fp = fname if os.path.isabs(fname) else os.path.join(RESULTS_DIR, fname)
        fp = os.path.abspath(fp)
        if not fp.startswith(os.path.abspath(RESULTS_DIR) + os.sep):
            raise HTTPException(status_code=400, detail=f"文件路径非法: {fname}")
        if not os.path.exists(fp):
            continue

        records = list(iter_jsonl_records(fp))
        by_index: Dict[int, Dict[str, Any]] = {}
        by_qimg: Dict[tuple, List[Dict[str, Any]]] = {}
        for rec in records:
            idx = rec.get("index")
            if isinstance(idx, int) and idx not in by_index:
                by_index[idx] = rec
            q = rec.get("question") or ""
            imgs = rec.get("images") or []
            key = (q, tuple(imgs))
            by_qimg.setdefault(key, []).append(rec)

        file_data[fname] = {"records": records, "by_index": by_index, "by_qimg": by_qimg}
    
    # 若所有文件都未加载到任何记录，则返回 404
    has_records = any(len((v.get("records") or [])) > 0 for v in file_data.values())
    if not has_records:
        raise HTTPException(status_code=404, detail="没有找到有效文件")
    
    # 按问题文本聚合（使用第一个文件的问题作为基准）
    base_file = file_list[0]
    base_records = file_data.get(base_file, {}).get("records", [])
    
    grouped_items = []
    for base_rec in base_records:
        base_q = base_rec.get("question", "")
        if query and not filter_match(base_rec, query):
            continue
        
        # 收集所有模型对此问题的回答（优先按 index 对齐，退化到 question+images 完整匹配）
        model_answers: Dict[str, Optional[Dict[str, Any]]] = {}
        base_idx = base_rec.get("index")
        base_imgs = base_rec.get("images", [])
        key_qimg = (base_q, tuple(base_imgs))

        for fname in file_list:
            data = file_data.get(fname, {})
            found = None
            # 1) 优先使用 index 精确对齐
            if isinstance(base_idx, int):
                found = (data.get("by_index") or {}).get(base_idx)
            # 2) 退化：使用 (question, images) 键匹配
            if found is None:
                matches = (data.get("by_qimg") or {}).get(key_qimg) or []
                if matches:
                    found = matches[0]
            # 3) 最后退化：完整 question 文本相等（不再做前缀匹配）
            if found is None:
                for rec in (data.get("records") or []):
                    if (rec.get("question") or "") == base_q:
                        found = rec
                        break

            if found is not None:
                model_answers[fname] = {
                    "answer": found.get("answer", ""),
                    "ground_truth": found.get("ground_truth"),
                    "reward": found.get("reward", {}),
                    "is_correct": is_correct(found),
                    "model_dir": found.get("model_dir", ""),
                }
            else:
                model_answers[fname] = None
        
        # 过滤正确性
        if correctness != "all":
            correct_values = [ma["is_correct"] for ma in model_answers.values() if ma]
            if correctness == "true" and not any(c is True for c in correct_values):
                continue
            if correctness == "false" and not any(c is False for c in correct_values):
                continue
            if correctness == "mixed" and len(set(correct_values)) <= 1:
                continue
        
        grouped_items.append({
            "index": base_rec.get("index"),
            "question": base_q,
            "images": base_rec.get("images", []),
            "models": model_answers,
        })
    
    # 分页
    total = len(grouped_items)
    page_items = grouped_items[offset: offset + limit]
    
    # 计算汇总统计
    summary_by_model = {}
    for fname in file_list:
        model_results = [item["models"].get(fname) for item in grouped_items if item["models"].get(fname)]
        correct = sum(1 for r in model_results if r and r.get("is_correct") is True)
        total_model = len(model_results)
        with_scores = [r for r in model_results if r and isinstance(r.get("reward", {}).get("total"), (int, float))]
        avg_reward = (sum(r["reward"]["total"] for r in with_scores) / len(with_scores)) if with_scores else 0.0
        
        summary_by_model[fname] = {
            "total": total_model,
            "correct": correct,
            "accuracy": (correct / total_model) if total_model > 0 else 0.0,
            "avg_total_reward": avg_reward,
        }
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page_items,
        "summary_by_model": summary_by_model,
        "files": file_list,
    }



