import asyncio
import re
import logging
import time
from typing import List, Dict, Optional, Tuple
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AsyncOpenAI
import json

class DomainClassifier:
    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        embeddings_service,
        chatgpt_deployment: str
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.embeddings_service = embeddings_service
        self.chatgpt_deployment = chatgpt_deployment
        
        # Enhanced caching for performance
        self._static_keywords_cache = None
        self._dynamic_keywords_cache = {}
        self._cache_timestamp = None
        self._initialization_done = False
        
    async def classify_with_context(
        self, 
        question: str, 
        conversation_history: List[Dict] = None
    ) -> Tuple[List[str], float, str]:
        """
        Ultra-fast classification with aggressive early returns
        Returns: (domains, confidence, reasoning)
        """
        start_time = time.time()
        
        # Step 1: FASTEST - Static keyword pre-classification (no I/O)
        static_result = self._check_static_keywords_fast(question)
        if static_result:
            elapsed = (time.time() - start_time) * 1000
            domains, confidence, reasoning = static_result
            logging.debug(f"⚡ Static keyword match in {elapsed:.1f}ms: {reasoning}")
            return (domains, confidence, f"Fast static match: {reasoning}")
        
        # Step 2: FAST - Dynamic keyword check (cached, minimal I/O)
        dynamic_result = await self._check_dynamic_keywords_fast(question)
        if dynamic_result:
            elapsed = (time.time() - start_time) * 1000
            domains, confidence, reasoning = dynamic_result
            logging.debug(f"⚡ Dynamic keyword match in {elapsed:.1f}ms: {reasoning}")
            return (domains, confidence, f"Fast dynamic match: {reasoning}")
        
        # Step 3: FALLBACK - Vector search only when necessary (expensive)
        logging.debug(f"🔍 No keyword match, falling back to vector search...")
        vector_result = await self._vector_search_classification(question)
        elapsed = (time.time() - start_time) * 1000
        logging.debug(f"📊 Vector classification completed in {elapsed:.1f}ms")
        return vector_result
    
    def _check_static_keywords_fast(self, question: str) -> Optional[Tuple[List[str], float, str]]:
        """Ultra-fast static keyword matching - no I/O operations"""
        question_lower = question.lower()
        
        # IMMEDIATE HIGH CONFIDENCE - Explicit domain mentions
        cosmic_explicit = ['cosmic cli', 'cosmic container', 'cosmic-prod', 'cosmic platform', 'cosmic windows', 'cosmic linux']
        substrate_explicit = ['substrate infrastructure', 'substrate platform', 'substrate deployment', 'azsc', 'azure service catalog']
        
        for term in cosmic_explicit:
            if term in question_lower:
                return (["Cosmic"], 0.98, f"Explicit Cosmic mention: '{term}'")
        
        for term in substrate_explicit:
            if term in question_lower:
                return (["Substrate"], 0.98, f"Explicit Substrate mention: '{term}'")
        
        # HIGH CONFIDENCE - Strong domain signals
        cosmic_strong = [
            'ep portal', 'watson', 'cosmic', 'container memory dump', 
            'container performance', 'container diagnostics', 'profiling tool',
            'cosmic-prod-', 'watson profiling'
        ]
        
        substrate_strong = [
            'exchange server', 'exchange performance', 'azscwork@microsoft.com',
            'infrastructure deployment', 'server management', 'platform deployment'
        ]
        
        cosmic_matches = [term for term in cosmic_strong if term in question_lower]
        substrate_matches = [term for term in substrate_strong if term in question_lower]
        
        # Single domain with strong signal
        if cosmic_matches and not substrate_matches:
            return (["Cosmic"], 0.95, f"Strong Cosmic signal: '{cosmic_matches[0]}'")
        
        if substrate_matches and not cosmic_matches:
            return (["Substrate"], 0.95, f"Strong Substrate signal: '{substrate_matches[0]}'")
        
        # CONTEXT-AWARE CROSS-DOMAIN DETECTION
        return self._check_contextual_terms_fast(question_lower)
    
    def _check_contextual_terms_fast(self, question_lower: str) -> Optional[Tuple[List[str], float, str]]:
        """Fast context-aware detection for ambiguous terms"""
        
        # Cross-domain diagnostic terms
        diagnostic_terms = ['memory dump', 'performance issue', 'troubleshooting', 'diagnostic', 'monitoring']
        diagnostic_matches = [term for term in diagnostic_terms if term in question_lower]
        
        if diagnostic_matches:
            # Quick context detection
            container_indicators = ['container', 'docker', 'kubernetes', 'pod']
            server_indicators = ['server', 'exchange', 'infrastructure', 'vm', 'virtual machine']
            
            has_container_context = any(term in question_lower for term in container_indicators)
            has_server_context = any(term in question_lower for term in server_indicators)
            
            # Clear context = single domain
            if has_container_context and not has_server_context:
                return (["Cosmic"], 0.8, f"Diagnostic term with container context: '{diagnostic_matches[0]}'")
            elif has_server_context and not has_container_context:
                return (["Substrate"], 0.8, f"Diagnostic term with server context: '{diagnostic_matches[0]}'")
            elif not has_container_context and not has_server_context:
                # No context = multi-domain for safety
                return (["Cosmic", "Substrate"], 0.6, f"Ambiguous diagnostic term: '{diagnostic_matches[0]}'")
        
        return None
    
    async def _check_dynamic_keywords_fast(self, question: str) -> Optional[Tuple[List[str], float, str]]:
        """Fast dynamic keyword check with aggressive caching"""
        
        # Initialize dynamic keywords cache if needed (only once)
        if not self._initialization_done:
            await self._initialize_dynamic_keywords()
        
        if not self._dynamic_keywords_cache:
            return None
        
        question_lower = question.lower()
        cosmic_keywords = self._dynamic_keywords_cache.get("Cosmic", {})
        substrate_keywords = self._dynamic_keywords_cache.get("Substrate", {})
        
        # Fast scoring without complex fuzzy matching
        cosmic_score = self._fast_keyword_score(question_lower, cosmic_keywords)
        substrate_score = self._fast_keyword_score(question_lower, substrate_keywords)
        
        if cosmic_score == 0 and substrate_score == 0:
            return None
        
        max_score = max(cosmic_score, substrate_score)
        score_diff = abs(cosmic_score - substrate_score)
        
        # Decisive thresholds for fast classification
        if max_score >= 0.8 and score_diff >= 0.3:
            if cosmic_score > substrate_score:
                return (["Cosmic"], 0.9, f"Clear dynamic keyword match for Cosmic (score: {cosmic_score:.2f})")
            else:
                return (["Substrate"], 0.9, f"Clear dynamic keyword match for Substrate (score: {substrate_score:.2f})")
        
        elif max_score >= 0.6 and score_diff >= 0.2:
            if cosmic_score > substrate_score:
                return (["Cosmic"], 0.8, f"Good dynamic keyword match for Cosmic (score: {cosmic_score:.2f})")
            else:
                return (["Substrate"], 0.8, f"Good dynamic keyword match for Substrate (score: {substrate_score:.2f})")
        
        return None
    
    def _fast_keyword_score(self, question: str, domain_keywords: Dict) -> float:
        """Ultra-fast keyword scoring - exact matches only"""
        if not domain_keywords:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # High confidence keywords (weight: 1.0)
        high_conf = domain_keywords.get("high_confidence", [])
        for keyword in high_conf:
            total_weight += 1.0
            if keyword.lower() in question:
                score += 1.0
        
        # Medium confidence keywords (weight: 0.6)
        medium_conf = domain_keywords.get("medium_confidence", [])
        for keyword in medium_conf:
            total_weight += 0.6
            if keyword.lower() in question:
                score += 0.6
        
        # Extracted keywords (weight: 0.3)
        extracted = domain_keywords.get("extracted", [])
        for keyword in extracted:
            total_weight += 0.3
            if keyword.lower() in question:
                score += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    async def _initialize_dynamic_keywords(self):
        """One-time initialization of dynamic keywords"""
        try:
            logging.info("🔄 Initializing dynamic keywords cache...")
            start_time = time.time()
            
            # Single search call to get all domain data
            results = await self.search_client.search(
                search_text="*",
                select=["domain", "keywords", "topics"],
                top=10
            )
            
            keywords_by_domain = {}
            
            async for page in results.by_page():
                async for result in page:
                    domain = result.get("domain", "")
                    keywords_str = result.get("keywords", "")
                    topics_str = result.get("topics", "")
                    
                    if domain and (keywords_str or topics_str):
                        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
                        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
                        
                        domain_keywords = self._categorize_keywords_fast(domain, keywords, topics)
                        keywords_by_domain[domain] = domain_keywords
            
            self._dynamic_keywords_cache = keywords_by_domain
            self._cache_timestamp = time.time()
            self._initialization_done = True
            
            elapsed = (time.time() - start_time) * 1000
            logging.info(f"✅ Dynamic keywords initialized in {elapsed:.1f}ms for domains: {list(keywords_by_domain.keys())}")
            
        except Exception as e:
            logging.warning(f"Failed to initialize dynamic keywords: {e}")
            self._initialization_done = True  # Prevent retries
    
    def _categorize_keywords_fast(self, domain: str, keywords: List[str], topics: List[str]) -> Dict:
        """Fast keyword categorization"""
        high_confidence = []
        medium_confidence = []
        extracted = []
        
        if domain.lower() == "cosmic":
            high_patterns = ["cosmic", "watson", "ep portal", "container", "profiling"]
            medium_patterns = ["performance", "monitoring", "diagnostic", "telemetry"]
        elif domain.lower() == "substrate":
            high_patterns = ["substrate", "azsc", "azure service catalog", "exchange", "infrastructure"]
            medium_patterns = ["deployment", "platform", "management", "server"]
        else:
            high_patterns = []
            medium_patterns = []
        
        for keyword in keywords + topics:
            keyword_lower = keyword.lower()
            
            if any(pattern in keyword_lower for pattern in high_patterns):
                high_confidence.append(keyword)
            elif any(pattern in keyword_lower for pattern in medium_patterns):
                medium_confidence.append(keyword)
            else:
                extracted.append(keyword)
        
        return {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "extracted": extracted
        }
    
    async def _vector_search_classification(self, question: str) -> Tuple[List[str], float, str]:
        """Vector search as last resort - expensive but accurate"""
        try:
            start_time = time.time()
            
            # Create embedding (expensive operation)
            question_embedding = await self.embeddings_service.create_embeddings([question])
            embedding_time = (time.time() - start_time) * 1000
            
            # Vector search
            vector_query = VectorizedQuery(
                vector=question_embedding[0],
                k_nearest_neighbors=2,
                fields="embedding"
            )
            
            results = await self.search_client.search(
                search_text=question,
                vector_queries=[vector_query],
                select=["domain", "keywords", "topics"],
                top=2
            )
            
            # Get domain scores
            cosmic_score = 0.0
            substrate_score = 0.0
            
            async for page in results.by_page():
                async for result in page:
                    domain = result.get("domain", "").lower()
                    score = result.get("@search.score", 0.0)
                    
                    if domain == "cosmic":
                        cosmic_score = max(cosmic_score, score)
                    elif domain == "substrate":
                        substrate_score = max(substrate_score, score)
            
            total_time = (time.time() - start_time) * 1000
            logging.debug(f"📊 Vector search: embedding={embedding_time:.1f}ms, total={total_time:.1f}ms")
            
            # Fast vector classification with aggressive thresholds
            max_score = max(cosmic_score, substrate_score)
            score_diff = abs(cosmic_score - substrate_score)
            
            if max_score >= 0.7 and score_diff >= 0.1:
                if cosmic_score > substrate_score:
                    return (["Cosmic"], 0.85, f"Vector similarity to Cosmic (score: {cosmic_score:.3f})")
                else:
                    return (["Substrate"], 0.85, f"Vector similarity to Substrate (score: {substrate_score:.3f})")
            
            elif max_score >= 0.5 and score_diff >= 0.05:
                if cosmic_score > substrate_score:
                    return (["Cosmic"], 0.7, f"Moderate vector similarity to Cosmic (score: {cosmic_score:.3f})")
                else:
                    return (["Substrate"], 0.7, f"Moderate vector similarity to Substrate (score: {substrate_score:.3f})")
            
            else:
                return (["Cosmic", "Substrate"], 0.5, f"Ambiguous vector scores (C:{cosmic_score:.3f}, S:{substrate_score:.3f})")
                
        except Exception as e:
            logging.warning(f"Vector search failed: {e}")
            return (["Cosmic", "Substrate"], 0.4, f"Vector search failed, using both domains")
    
    # Keep LLM classification as backup (unchanged)
    async def classify_with_llm_fallback(
        self, 
        question: str, 
        conversation_history: List[Dict] = None
    ) -> Tuple[List[str], float, str]:
        """LLM classification fallback for complex cases"""
        # Implementation unchanged - only used when explicitly called
        pass