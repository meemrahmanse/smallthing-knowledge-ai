from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Any


class SearchStep(Enum):
    KEYWORDSEARCH = auto()
    SEMANTICSEARCH = auto()
    FILTERBYSOURCE = auto()
    AGGREGATE = auto()
    RERANK = auto()
    EXPANDCONTEXT = auto()


@dataclass
class PlanStep:
    steptype: SearchStep
    params: dict = field(default_factory=dict)
    dependson: List[str] = field(default_factory=list)
    stepid: str = ""
    estimatedtimems: int = 100


@dataclass
class SearchPlan:
    queryid: str
    analyzedquery: Any
    steps: List[PlanStep] = field(default_factory=list)
    estimatedtimems: int = 0


@dataclass
class SearchResult:
    chunks: List[dict]
    totalfound: int
    searchtimems: int
    strategyused: str
    sources: List[str]


class SearchPlanner:
    def __init__(self, rag_engine: Any = None, keyword_index: Any = None, reranker: Any = None):
        self.rag = rag_engine
        self.keyword_index = keyword_index
        self.reranker = reranker

    def create_plan(self, analyzed: Any) -> SearchPlan:
        import secrets
        queryid = secrets.token_hex(8)
        steps = []

        strat = analyzed.searchstrategy
        if strat == "keywordfirst":
            steps.append(PlanStep(SearchStep.KEYWORDSEARCH, {"codes": analyzed.codes, "queries": analyzed.rewrittenqueries}, stepid="kwprimary", estimatedtimems=50))
            steps.append(PlanStep(SearchStep.SEMANTICSEARCH, {"query": analyzed.original, "k": max(1, analyzed.suggestedk // 2)}, stepid="semsecondary", estimatedtimems=150))
        elif strat == "semanticfirst":
            steps.append(PlanStep(SearchStep.SEMANTICSEARCH, {"query": analyzed.original, "k": analyzed.suggestedk}, stepid="semprimary", estimatedtimems=150))
        elif strat == "exhaustive":
            steps.append(PlanStep(SearchStep.KEYWORDSEARCH, {"codes": analyzed.codes, "queries": analyzed.rewrittenqueries, "exhaustive": True}, stepid="kwexhaustive", estimatedtimems=200))
            steps.append(PlanStep(SearchStep.SEMANTICSEARCH, {"query": analyzed.original, "k": min(analyzed.suggestedk * 2, 50)}, stepid="semexhaustive", estimatedtimems=300))
        else:
            steps.append(PlanStep(SearchStep.KEYWORDSEARCH, {"codes": analyzed.codes, "queries": analyzed.rewrittenqueries}, stepid="kwhybrid", estimatedtimems=50))
            steps.append(PlanStep(SearchStep.SEMANTICSEARCH, {"query": analyzed.original, "k": analyzed.suggestedk}, stepid="semhybrid", estimatedtimems=150))

        if getattr(analyzed, 'locations', None):
            steps.append(PlanStep(SearchStep.FILTERBYSOURCE, {"locations": analyzed.locations}, dependson=[s.stepid for s in steps], stepid="filterlocation", estimatedtimems=20))

        steps.append(PlanStep(SearchStep.AGGREGATE, {"dedup": True}, dependson=[s.stepid for s in steps if not s.dependson], stepid="aggregate", estimatedtimems=30))
        steps.append(PlanStep(SearchStep.RERANK, {"topk": min(analyzed.suggestedk, 15)}, dependson=["aggregate"], stepid="rerank", estimatedtimems=100))

        if analyzed.complexity.name in ("COMPLEX", "EXPERT"):
            steps.append(PlanStep(SearchStep.EXPANDCONTEXT, {"window": 2}, dependson=["rerank"], stepid="expand", estimatedtimems=50))

        tot = sum(s.estimatedtimems for s in steps)
        return SearchPlan(queryid=queryid, analyzedquery=analyzed, steps=steps, estimatedtimems=tot)

    async def execute_plan(self, plan: SearchPlan) -> SearchResult:
        start = time.time()
        results_by_step = {}
        final_chunks = []

        for step in plan.steps:
            # ensure dependencies
            for dep in step.dependson:
                if dep not in results_by_step:
                    raise RuntimeError(f"Missing dependency {dep}")

            res = await self.execute_step(step, results_by_step)
            results_by_step[step.stepid] = res
            if isinstance(res, list):
                final_chunks = res

        elapsed = int((time.time() - start) * 1000)
        sources = list({c.get('source', 'unknown') for c in final_chunks})
        return SearchResult(chunks=final_chunks, totalfound=len(final_chunks), searchtimems=elapsed, strategyused=plan.analyzedquery.searchstrategy, sources=sources)

    async def execute_step(self, step: PlanStep, previousresults: dict) -> Any:
        if step.steptype == SearchStep.KEYWORDSEARCH:
            return self.keyword_search(step.params)
        if step.steptype == SearchStep.SEMANTICSEARCH:
            return self.semantic_search(step.params)
        if step.steptype == SearchStep.FILTERBYSOURCE:
            allchunks = []
            for dep in step.dependson:
                if dep in previousresults:
                    allchunks.extend(previousresults[dep])
            return self.filter_by_location(allchunks, step.params.get('locations', []))
        if step.steptype == SearchStep.AGGREGATE:
            allchunks = []
            for dep in step.dependson:
                if dep in previousresults:
                    allchunks.extend(previousresults[dep])
            return self.aggregate(allchunks)
        if step.steptype == SearchStep.RERANK:
            chunks = previousresults.get(step.dependson[0], []) if step.dependson else []
            return self.rerank(chunks, step.params)
        if step.steptype == SearchStep.EXPANDCONTEXT:
            chunks = previousresults.get(step.dependson[0], []) if step.dependson else []
            return chunks
        return []

    def keyword_search(self, params: dict) -> List[dict]:
        results = []
        codes = params.get('codes', [])
        if self.keyword_index and codes:
            for code in codes:
                for cid, score in self.keyword_index.search(code):
                    results.append({'text': self.keyword_index.getchunk(cid), 'source': self.keyword_index.getsource(cid), 'similarity': score, 'matchtype': 'keyword'})
        return results

    def semantic_search(self, params: dict) -> List[dict]:
        if self.rag:
            return self.rag.retrieve(params.get('query', ''), k=params.get('k', 10))
        return []

    def filter_by_location(self, chunks: List[dict], locations: List[str]) -> List[dict]:
        if not locations:
            return chunks
        filtered = []
        for c in chunks:
            up = c.get('text', '').upper()
            if any(loc in up for loc in locations):
                filtered.append(c)
        return filtered if filtered else chunks

    def aggregate(self, chunks: List[dict]) -> List[dict]:
        seen = set()
        out = []
        for c in chunks:
            h = hash((c.get('text', '')[:200]))
            if h not in seen:
                seen.add(h)
                out.append(c)
        return out

    def rerank(self, chunks: List[dict], params: dict) -> List[dict]:
        if self.reranker:
            return self.reranker.rerank(params.get('query', ''), chunks, topk=params.get('topk', 10))
        return chunks[:params.get('topk', 10)]
