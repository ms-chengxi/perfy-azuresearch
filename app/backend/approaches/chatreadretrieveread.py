import os  # Add this import at the top
from collections.abc import Awaitable
from typing import Any, AsyncGenerator, List, Optional, Union, cast

from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from approaches.approach import DataPoints, ExtraInfo, ThoughtStep
from approaches.chatapproach import ChatApproach
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        search_client: SearchClient,
        search_index_name: str,
        agent_model: Optional[str],
        agent_deployment: Optional[str],
        agent_client: Optional[KnowledgeAgentRetrievalClient],
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],
        embedding_deployment: Optional[str],
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
        domain_classifier: Optional[Any] = None,
        openai_host: str = "",
        vision_endpoint: str = "",
        vision_token_provider: Optional[Any] = None,
    ):
        # Call parent class __init__ with the correct parameters
        super().__init__(
            search_client=search_client,
            openai_client=openai_client,
            auth_helper=auth_helper,
            query_language=query_language,
            query_speller=query_speller,
            embedding_deployment=embedding_deployment,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_field=embedding_field,
            openai_host=openai_host,
            vision_endpoint=vision_endpoint,
            vision_token_provider=vision_token_provider or (lambda: ""),
            prompt_manager=prompt_manager,
            reasoning_effort=reasoning_effort,
        )
        
        # Set additional attributes specific to this class
        self.search_index_name = search_index_name  # Store it as instance variable
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.agent_model = agent_model
        self.agent_deployment = agent_deployment
        self.agent_client = agent_client
        self.query_rewrite_prompt = prompt_manager.load_prompt("chat_query_rewrite.prompty")
        self.query_rewrite_tools = prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = prompt_manager.load_prompt("chat_answer_question.prompty")
        self.include_token_usage = True
        self.domain_classifier = domain_classifier

        # Add missing index names for agentic retrieval
        self.cosmic_index_name = os.getenv("AZURE_SEARCH_COSMIC_INDEX", "cosmic-index")
        self.substrate_index_name = os.getenv("AZURE_SEARCH_SUBSTRATE_INDEX", "substrate-index")

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]]]:
        use_agentic_retrieval = True if overrides.get("use_agentic_retrieval") else False
        original_user_query = messages[-1]["content"]

        # Get domain classification if classifier is available
        domain_info = None
        domain_message = ""
        if self.domain_classifier is not None:
            # Extract history from messages for classification
            history = []
            for msg in messages[:-1]:
                if msg.get("role") and msg.get("content"):
                    history.append({"role": msg["role"], "content": msg["content"]})
            
            domains, confidence, reasoning = await self.domain_classifier.classify_with_context(
                original_user_query, 
                history
            )
            print("🔍 Domain Classification Result:")
            print(f"   Domains: {domains}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Reasoning: {reasoning}")
            domain_message = self._format_domain_message(domains, confidence, reasoning)
            domain_info = {
                "domains": domains,
                "confidence": confidence,
                "reasoning": reasoning,
                "message": domain_message
            }
            
            # Store domain info in overrides so search approaches can use it
            overrides["domain_classification"] = domain_info
            
            print(f"🎯 Domain Classification Result:")
            print(f"   Domains: {domains}")
            print(f"   Domain count: {len(domains)}")
            print(f"   Message: {domain_message[:100]}...")

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )
        
        if use_agentic_retrieval and self.agent_client is not None:
            extra_info = await self.run_agentic_retrieval_approach(messages, overrides, auth_claims)
        else:
            extra_info = await self.run_search_approach(messages, overrides, auth_claims)

        # Get base prompt variables
        base_prompt_vars = self.get_system_prompt_variables(overrides.get("prompt_template"))
        
        # Check if override_prompt exists
        if "override_prompt" in base_prompt_vars and base_prompt_vars["override_prompt"]:
            print(f"⚠️ WARNING: override_prompt is set, domain formatting may not work")

        # Build prompt variables
        prompt_variables = base_prompt_vars | {
            "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
            "past_messages": messages[:-1],
            "user_query": original_user_query,
            "text_sources": extra_info.data_points.text,
            "injected_prompt": base_prompt_vars.get("injected_prompt", ""),
            "override_prompt": base_prompt_vars.get("override_prompt", ""),
        }
        
        # ALWAYS set domain variables (even if empty)
        prompt_variables["domain_prefix"] = ""
        prompt_variables["domain_context"] = ""
        print("domain info before adding to prompt variables:")
        print(f"Domain info: {domain_info}")
        # Add domain-specific variables if available
        if domain_info:
            # Set domain prefix (classification message)
            if domain_message:
                prompt_variables["domain_prefix"] = domain_message
                print(f"✅ Set domain_prefix: {domain_message[:50]}...")
            
            # Set domain context for multi-domain responses OR medium confidence (confusion category)
            domains_list = domain_info["domains"]
            confidence = domain_info["confidence"]
            
            # Treat as multi-domain if: 
            # 1. Actually multiple domains, OR 
            # 2. Medium confidence (0.5-0.7) indicating confusion/ambiguity
            should_use_multi_domain = (
                len(domains_list) > 1 or 
                (0.5 <= confidence <= 0.7)
            )
            
            if should_use_multi_domain:
                print(f"🔀 Triggering multi-domain formatting:")
                print(f"   Reason: {'Multiple domains' if len(domains_list) > 1 else 'Medium confidence (confusion category)'}")
                print(f"   Domains: {domains_list}")
                print(f"   Confidence: {confidence}")
                
                # For medium confidence with single domain, add both domains for comparison
                if len(domains_list) == 1 and 0.5 <= confidence <= 0.7:
                    # Add the "other" domain for comparison since we're uncertain
                    if "Cosmic" in domains_list:
                        domains_list = ["Cosmic", "Substrate"]
                    elif "Substrate" in domains_list:
                        domains_list = ["Cosmic", "Substrate"]
                    print(f"   Expanded domains for confusion category: {domains_list}")
                
                # Build explicit formatting instructions with better separation
                domain_sections = []
                for domain in domains_list:
                    if domain == "Cosmic":
                        domain_sections.append("### Cosmic:\n\n[All Cosmic-related information from sources tagged with 'Domain: Cosmic']\n\n[Include citations [source.docx] for each fact]")
                    elif domain == "Substrate":
                        domain_sections.append("### Substrate:\n\n[All Exchange/Substrate information from sources tagged with 'Domain: Substrate']\n\n[Include citations [source.docx] for each fact]")
                    else:
                        domain_sections.append(f"### {domain}:\n\n[All {domain}-related information]\n\n[Include citations [source.docx] for each fact]")
                
                confusion_note = ""
                if 0.5 <= confidence <= 0.7:
                    confusion_note = (
                        f"\n\nNOTE: The classification confidence was medium ({confidence:.1%}), "
                        f"so I'm providing information from both domains to ensure completeness."
                    )
                
                domain_context = (
                    f"This question involves {len(domains_list)} domains: {', '.join(domains_list)}.\n"
                    f"YOU MUST structure your answer with EXACTLY {len(domains_list)} sections.\n\n"
                    f"FORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n\n"
                    + "\n\n\n\n".join(domain_sections) + "\n\n"  # Four newlines between sections
                    f"\nCRITICAL REQUIREMENTS:\n"
                    f"- Each section MUST start with the ### heading shown above\n"
                    f"- Include citations [source.docx] for each fact\n"
                    f"- Add FOUR blank lines between domain sections for clear separation\n"
                    f"- DO NOT combine domains. DO NOT skip sections. DO NOT mix information.\n"
                    f"- If no information is available for a domain, state 'No specific information available for this domain.'\n"
                    f"- End each domain section with a period before starting the next section"
                    + confusion_note
                )
                
                prompt_variables["domain_context"] = domain_context
                print(f"✅ Set domain_context for {len(domains_list)} domains (confidence: {confidence:.1%})")
                print(f"   Context length: {len(domain_context)}")
                print(f"   Context preview: {domain_context[:200]}...")
            else:
                print(f"🔸 Single domain, high confidence - using standard formatting")
        
        # Debug: Print final prompt variables
        print(f"\n🔧 Final prompt variables:")
        for key, value in prompt_variables.items():
            if key in ["domain_prefix", "domain_context", "injected_prompt", "override_prompt"]:
                if isinstance(value, str):
                    print(f"   {key}: '{value[:100]}...' (length: {len(value)})")
                else:
                    print(f"   {key}: {value}")
            elif key == "text_sources":
                print(f"   {key}: {len(value)} sources")
            else:
                print(f"   {key}: {type(value).__name__}")

        # Render the prompt
        print("rendering prompt with variables:")
        print(prompt_variables["domain_prefix"] if "domain_prefix" in prompt_variables else "No domain prefix")
        print(prompt_variables["domain_context"] if "domain_context" in prompt_variables else "No domain context")
        messages = self.prompt_manager.render_prompt(
            self.answer_prompt,
            prompt_variables
        )
        
        # Debug: Check rendered message
        if messages and len(messages) > 0:
            system_msg = messages[0].get("content", "")
            print(f"\n🔍 Rendered system message check:")
            print(f"   Contains domain_prefix: {'📚 Based on your question' in system_msg}")
            print(f"   Contains domain_context: {'CRITICAL FORMATTING REQUIREMENT' in system_msg}")
            print(f"   Contains Under Cosmic: {'Under Cosmic:' in system_msg}")
            print(f"   System message length: {len(system_msg)}")
            
            # Print first 500 chars to see what's actually there
            print(f"\n📝 System message preview:")
            print(system_msg[:500])

        chat_coroutine = cast(
            Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]],
            self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                self.get_response_token_limit(self.chatgpt_model, 1024),
                should_stream,
            ),
        )
        
        extra_info.thoughts.append(
            self.format_thought_step_for_chatcompletion(
                title="Prompt to generate answer",
                messages=messages,
                overrides=overrides,
                model=self.chatgpt_model,
                deployment=self.chatgpt_deployment,
                usage=None,
            )
        )
        
        # Add domain classification as a thought step
        if domain_info:
            extra_info.thoughts.insert(0, ThoughtStep(
                title="Domain Classification",
                description=domain_info["message"],
                props={
                    "domains": domain_info["domains"],
                    "confidence": domain_info["confidence"],
                    "reasoning": domain_info["reasoning"],
                    "classifier": "DomainClassifier"
                }
            ))
        
        return (extra_info, chat_coroutine)

    async def run_search_approach(
        self, messages: list[ChatCompletionMessageParam], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ):
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        
        # Build base filter (security, permissions, etc.)
        base_search_filter = self.build_filter(overrides, auth_claims)

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        # Determine which index to use and build domain-specific filter
        domain_info = overrides.get("domain_classification")
        search_index_name = self.search_index_name  # Use the dedicated index for this approach
        search_index_filter = base_search_filter  # Start with base filter
        
        if domain_info and domain_info.get("domains"):
            domains = domain_info["domains"]
            if len(domains) == 1:
                domain = domains[0]
                # Add category filter if needed (for mixed-domain indexes)
                domain_category = overrides.get("include_category")
                if domain_category:
                    # Handle both string and list for include_category
                    if isinstance(domain_category, list):
                        # Multiple categories specified - add parentheses for proper OData syntax
                        category_filters = [f"(category eq '{cat}')" for cat in domain_category]
                        categories_filter = " or ".join(category_filters)
                        category_filter = f"({categories_filter})"
                    else:
                        # Single category specified
                        category_filter = f"(category eq '{domain_category}')"
                    
                    if base_search_filter:
                        search_index_filter = f"({base_search_filter}) and {category_filter}"
                    else:
                        search_index_filter = category_filter
                    print(f"🎯 Using domain filter for {domain_category}: {category_filter}")
                        
            else:
                # Multiple domains - add category filter for all requested domains  
                domain_category = overrides.get("include_category")
                if domain_category:
                    # Use the explicit include_category if provided
                    if isinstance(domain_category, list):
                        categories_filter = " or ".join([f"category eq '{cat}'" for cat in domain_category])
                    else:
                        categories_filter = f"category eq '{domain_category}'"
                else:
                    # Fall back to using the classified domains
                    categories_filter = " or ".join([f"category eq '{domain}'" for domain in domains])
                
                if base_search_filter:
                    search_index_filter = f"({base_search_filter}) and ({categories_filter})"
                else:
                    search_index_filter = categories_filter
                print(f"🎯 Multiple domains detected: {domains}. Using filter: {search_index_filter}")

        query_messages = self.prompt_manager.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        # STEP 1: Generate search query
        chat_completion = cast(
            ChatCompletion,
            await self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages=query_messages,
                overrides=overrides,
                response_token_limit=self.get_response_token_limit(self.chatgpt_model, 100),
                temperature=0.0,
                tools=tools,
                reasoning_effort="low",
            ),
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: Search with domain-aware filter
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            search_index_filter,  # Now includes domain filtering
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
            use_query_rewriting,
        )
        
        # Log what filter was used
        print(f"🔍 Search filter applied: {search_index_filter}")

        # STEP 3: Process results
        text_sources = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        
        # Add domain tagging to sources if domain is classified
        if domain_info and domain_info.get("domains"):
            domains = domain_info["domains"]
            if len(domains) == 1:
                domain = domains[0]
                tagged_sources = []
                for source in text_sources:
                    tagged_sources.append(f"[Domain: {domain}] {source}")
                text_sources = tagged_sources

        extra_info = ExtraInfo(
            DataPoints(text=text_sources),
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=query_messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=chat_completion.usage,
                    reasoning_effort="low",
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "use_query_rewriting": use_query_rewriting,
                        "top": top,
                        "filter": search_index_filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                        "search_index": search_index_name,
                        "domain_filter_applied": domain_info is not None,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
            ],
        )
        return extra_info

    async def run_agentic_retrieval_approach(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
    ):
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
        top = overrides.get("top", 3)
        max_subqueries = overrides.get("max_subqueries", 10)
        results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
        max_docs_for_reranker = max_subqueries * 50

        # Build base filter (security, permissions, etc.)
        base_search_filter = self.build_filter(overrides, auth_claims)

        # Determine which indexes to search based on domain classification
        domain_info = overrides.get("domain_classification")
        indexes_to_search = []
        
        if domain_info and domain_info.get("domains"):
            domains = domain_info["domains"]
            if "Cosmic" in domains:
                indexes_to_search.append(self.cosmic_index_name)
            if "Substrate" in domains:
                indexes_to_search.append(self.substrate_index_name)
        else:
            # No domain classification - search default index
            indexes_to_search.append(self.search_index_name)
        
        # If multiple indexes, perform parallel searches
        if len(indexes_to_search) > 1:
            print(f"🎯 Searching multiple indexes: {indexes_to_search}")
            
            # Run searches in parallel for each index
            import asyncio
            search_tasks = []
            domain_mapping = {}
            
            for index_name in indexes_to_search:
                # Track which domain each index corresponds to
                if index_name == self.cosmic_index_name:
                    domain_mapping[index_name] = "Cosmic"
                elif index_name == self.substrate_index_name:
                    domain_mapping[index_name] = "Substrate"
                else:
                    domain_mapping[index_name] = "Default"
                
                # Build index-specific filter
                search_index_filter = base_search_filter
                if domain_mapping[index_name] != "Default":
                    # Add category filter for this specific domain
                    category_filter = f"category eq '{domain_mapping[index_name]}'"
                    if base_search_filter:
                        search_index_filter = f"({base_search_filter}) and ({category_filter})"
                    else:
                        search_index_filter = category_filter
                    print(f"🔍 Filter for {domain_mapping[index_name]} index: {search_index_filter}")
                    
                task = self.run_agentic_retrieval(
                    messages=messages,
                    agent_client=self.agent_client,
                    search_index_name=index_name,
                    top=top,
                    filter_add_on=search_index_filter,  # Use domain-specific filter
                    minimum_reranker_score=minimum_reranker_score,
                    max_docs_for_reranker=max_docs_for_reranker,
                    results_merge_strategy=results_merge_strategy,
                )
                search_tasks.append((task, index_name))
            
            # Execute searches in parallel
            search_results_with_index = await asyncio.gather(*[task for task, _ in search_tasks])
            
            # Merge results from all indexes with domain tagging
            merged_response = search_results_with_index[0][0]  # Use first response as base
            merged_results = []
            
            for i, (response, results) in enumerate(search_results_with_index):
                index_name = search_tasks[i][1]
                domain = domain_mapping.get(index_name, "Unknown")
                
                # Tag each result with its source domain
                for result in results:
                    # Add domain tag to the content
                    if hasattr(result, 'content'):
                        result.content = f"[Domain: {domain}] {result.content}"
                    merged_results.append(result)
                    
                # Merge activity logs if needed
                if response.activity and merged_response.activity:
                    merged_response.activity.extend(response.activity)
            
            # Re-rank and deduplicate merged results
            # Sort by relevance score and take top N
            merged_results.sort(key=lambda r: r.score if hasattr(r, 'score') else 0, reverse=True)
            
            # Remove duplicates based on content similarity
            unique_results = []
            seen_content = set()
            for result in merged_results:
                content_key = result.content[:200] if hasattr(result, 'content') else str(result)
                if content_key not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_key)
                    if len(unique_results) >= top:
                        break
            
            response, results = merged_response, unique_results[:top]
            final_search_filter = "Multiple domain-specific filters applied"
            
        else:
            # Single index search
            search_index = indexes_to_search[0]
            
            # Build filter for single index search
            search_index_filter = base_search_filter
            domain_used = "Default"
            
            if domain_info and domain_info.get("domains"):
                domains = domain_info["domains"]
                if len(domains) == 1:
                    domain = domains[0]
                    if search_index == self.cosmic_index_name and domain == "Cosmic":
                        category_filter = "category eq 'Cosmic'"
                        domain_used = "Cosmic"
                    elif search_index == self.substrate_index_name and domain == "Substrate":
                        category_filter = "category eq 'Substrate'"
                        domain_used = "Substrate"
                    else:
                        category_filter = None
                    
                    if category_filter:
                        if base_search_filter:
                            search_index_filter = f"({base_search_filter}) and ({category_filter})"
                        else:
                            search_index_filter = category_filter
                else:
                    # Multiple domains on single index - use OR filter
                    categories_filter = " or ".join([f"category eq '{domain}'" for domain in domains])
                    if base_search_filter:
                        search_index_filter = f"({base_search_filter}) and ({categories_filter})"
                    else:
                        search_index_filter = categories_filter
                    domain_used = "Multiple"
            
            print(f"🎯 Using single index: {search_index}")
            print(f"🔍 Filter applied: {search_index_filter}")
            
            response, results = await self.run_agentic_retrieval(
                messages=messages,
                agent_client=self.agent_client,
                search_index_name=search_index,
                top=top,
                filter_add_on=search_index_filter,  # Use domain-specific filter
                minimum_reranker_score=minimum_reranker_score,
                max_docs_for_reranker=max_docs_for_reranker,
                results_merge_strategy=results_merge_strategy,
            )
            
            # Tag results for single domain if needed
            if domain_used != "Default" and domain_used != "Multiple":
                for result in results:
                    if hasattr(result, 'content'):
                        result.content = f"[Domain: {domain_used}] {result.content}"
            
            final_search_filter = search_index_filter

        text_sources = self.get_sources_content(results, use_semantic_captions=False, use_image_citation=False)

        extra_info = ExtraInfo(
            DataPoints(text=text_sources),
            thoughts=[
                ThoughtStep(
                    "Use agentic retrieval",
                    messages,
                    {
                        "reranker_threshold": minimum_reranker_score,
                        "max_docs_for_reranker": max_docs_for_reranker,
                        "results_merge_strategy": results_merge_strategy,
                        "filter": final_search_filter,
                        "indexes_searched": indexes_to_search,
                        "domain_filter_applied": domain_info is not None,
                    },
                ),
                ThoughtStep(
                    f"Agentic retrieval results (top {top})",
                    [result.serialize_for_results() for result in results],
                    {
                        "query_plan": (
                            [activity.as_dict() for activity in response.activity] if response.activity else None
                        ),
                        "model": self.agent_model,
                        "deployment": self.agent_deployment,
                    },
                ),
            ],
        )
        
        # Add domain classification as a thought step for agentic retrieval too
        if domain_info:
            extra_info.thoughts.insert(0, ThoughtStep(
                title="Domain Classification",
                description=domain_info["message"],
                props={
                    "domains": domain_info["domains"],
                    "confidence": domain_info["confidence"],
                    "reasoning": domain_info["reasoning"],
                    "classifier": "DomainClassifier"
                }
            ))
        
        return extra_info

    def _format_domain_message(self, domains: List[str], confidence: float, reasoning: str) -> str:
        """Format domain classification into user-friendly message"""
        if len(domains) == 1:
            domain = domains[0]
            if confidence >= 0.8:
                return f"📚 Based on your question, I believe this is related to the **{domain}** domain."
            elif confidence >= 0.5:
                return f"📚 Based on your question, I think this is related to the **{domain}** domain, though I'm not entirely certain."
            else:
                return f"📚 Based on your question, this might be related to the **{domain}** domain, but I'm not confident about this classification."
        else:
            # Multiple domains
            domains_str = " and ".join(f"**{d}**" for d in domains)
            if confidence >= 0.8:
                return f"📚 Based on your question, this appears to be related to both {domains_str} domains."
            else:
                return f"📚 Based on your question, this might be related to both {domains_str} domains. {reasoning}"
