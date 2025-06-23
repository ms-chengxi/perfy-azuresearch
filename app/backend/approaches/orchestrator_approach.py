import asyncio
import re
import logging
from typing import Any, Optional

from approaches.chatapproach import ChatApproach
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.approach import ThoughtStep, DataPoints, ExtraInfo  # Add DataPoints, ExtraInfo imports


class OrchestratorApproach(ChatApproach):
    """
    Orchestrator approach that uses domain classification to route questions
    to appropriate domain-specific approaches. Inherits streaming from ChatApproach.
    """

    def __init__(
        self,
        cosmic_approach: ChatReadRetrieveReadApproach,
        substrate_approach: ChatReadRetrieveReadApproach,
        domain_classifier: Any,
        openai_client: Any,
        prompt_manager: Any,
    ):
        self.cosmic_approach = cosmic_approach
        self.substrate_approach = substrate_approach
        self.domain_classifier = domain_classifier
        # Set required properties for ChatApproach base class
        self.openai_client = openai_client
        self.prompt_manager = prompt_manager
        self.chatgpt_model = cosmic_approach.chatgpt_model
        self.chatgpt_deployment = cosmic_approach.chatgpt_deployment
        # Add missing attributes that ChatApproach expects
        self.include_token_usage = True  # Enable token usage tracking

    def _get_original_filenames_from_sources(self, text_sources: list[str]) -> set[str]:
        """Extract original filenames from formatted source text by looking for sourcefile patterns"""
        filenames = set()
        for source in text_sources:
            # The source format is typically: "[filename]: content"
            # We want to extract the filename part before the colon
            try:
                # Remove domain tags first
                clean_source = re.sub(r'\[Domain:\s*\w+\]\s*', '', source)
                
                # Look for the citation pattern at the start: [filename.ext]:
                match = re.match(r'\[([^\]]+\.\w+)\]:', clean_source)
                if match:
                    filename = match.group(1)
                    filenames.add(filename)
                else:
                    # Fallback: look for any filename pattern in the source
                    filename_match = re.search(r'([^\s\[\]]+\.\w+)', clean_source)
                    if filename_match:
                        filenames.add(filename_match.group(1))
            except Exception as e:
                logging.warning(f"Error extracting filename from source: {e}")
                continue
        return filenames

    async def _close_unused_coroutine(self, coroutine, name: str):
        """Safely close an unused coroutine to prevent runtime warnings"""
        if coroutine is not None:
            try:
                print(f"🗑️ Closing unused {name} coroutine")
                coroutine.close()
            except Exception as e:
                logging.warning(f"Error closing {name} coroutine: {e}")

    async def run_until_final_call(
        self, messages, overrides, auth_claims, should_stream
    ):
        """
        Override the main logic - route to appropriate domain approach with enhanced prompting
        """
        # Check if user has explicitly selected a category
        user_category = overrides.get("include_category", "")
        
        if user_category and user_category != "":
            # User specified a category, use it directly with domain-specific prompting
            if user_category == "Cosmic":
                overrides["domain_prefix"] = "📚 Based on your question about Microsoft's Cosmic container platform:"
                overrides["domain_context"] = """You are an expert on Microsoft's Cosmic container platform for performance and diagnostics.

CRITICAL CITATION REQUIREMENTS:
- Provide comprehensive factual information from the sources
- DO NOT include inline citations like [filename.docx] within your response text
- At the end of your response, add a "## References:" section
- In the References section, list each source file as [filename.docx] on its own line
- Extract clean filenames without domain tags or URLs
- This format ensures citations will be clickable for users"""
                return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
            elif user_category == "Substrate":
                overrides["domain_prefix"] = "🏗️ Based on your question about Microsoft's Substrate infrastructure:"
                overrides["domain_context"] = """You are an expert on Microsoft's Substrate infrastructure platform.

CRITICAL CITATION REQUIREMENTS:
- Provide comprehensive factual information from the sources
- DO NOT include inline citations like [filename.docx] within your response text
- At the end of your response, add a "## References:" section
- In the References section, list each source file as [filename.docx] on its own line
- Extract clean filenames without domain tags or URLs
- This format ensures citations will be clickable for users"""
                return await self.substrate_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
        
        # Extract the latest user question
        user_question = messages[-1]["content"] if messages else ""
        
        # Get domain classification with context
        try:
            domains, confidence, reasoning = await self.domain_classifier.classify_with_context(
                user_question,
                messages[:-1]  # Pass conversation history
            )
            
            print(f"🎯 Orchestrator Classification Result:")
            print(f"   Question: {user_question}")
            print(f"   Domains: {domains}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Reasoning: {reasoning}")
            
        except Exception as e:
            print(f"❌ Domain classification error: {e}")
            # Fall back to cosmic domain
            domains = ["Cosmic"]
            confidence = 0.5
            reasoning = f"Classification failed: {str(e)}, defaulting to Cosmic"
        
        # Handle based on classification results
        if len(domains) == 1 and confidence >= 0.6:
            # Single domain with high confidence
            domain = domains[0]
            overrides["include_category"] = domain
            
            # Add domain-specific prompt context with citation formatting instructions
            if domain == "Cosmic":
                overrides["domain_prefix"] = f"📚 Based on your question about Microsoft's Cosmic (confidence: {confidence:.1%}):"
                overrides["domain_context"] = """You are an expert on Microsoft's Cosmic container platform for performance and diagnostics.

CRITICAL CITATION REQUIREMENTS:
- Provide comprehensive factual information from the sources
- DO NOT include inline citations like [filename.docx] within your response text
- At the end of your response, add a "## References:" section
- In the References section, list each source file as [filename.docx] on its own line
- Extract clean filenames without domain tags or URLs
- This format ensures citations will be clickable for users"""
            else:
                overrides["domain_prefix"] = f"🏗️ Based on your question about Microsoft's Substrate (confidence: {confidence:.1%}):"
                overrides["domain_context"] = """You are an expert on Microsoft's Substrate infrastructure platform.

CRITICAL CITATION REQUIREMENTS:
- Provide comprehensive factual information from the sources
- DO NOT include inline citations like [filename.docx] within your response text
- At the end of your response, add a "## References:" section
- In the References section, list each source file as [filename.docx] on its own line
- Extract clean filenames without domain tags or URLs
- This format ensures citations will be clickable for users"""
            
            approach = self.cosmic_approach if domain == "Cosmic" else self.substrate_approach
            extra_info, chat_coroutine = await approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
            
            # Create proper ThoughtStep object
            classification_step = ThoughtStep(
                title="Domain Classification",
                description=f"{domain} (Confidence: {confidence:.1%}) - {reasoning}",
                props={
                    "domain": domain,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
            )
            
            if hasattr(extra_info, 'thoughts') and extra_info.thoughts:
                extra_info.thoughts.insert(0, classification_step)
            else:
                extra_info.thoughts = [classification_step]
            
            return extra_info, chat_coroutine
            
        elif len(domains) >= 2 or (len(domains) == 1 and confidence < 0.6):
            # Multi-domain or low confidence - run both approaches in parallel
            print(f"🔄 Multi-domain detected, running parallel searches for domains: {domains}")
            
            search_domains = domains if len(domains) >= 2 else ["Cosmic", "Substrate"]
            
            # For streaming, we need to disable streaming on individual searches
            # and only enable it on the final unified response
            individual_should_stream = False  # Always false for parallel searches
            
            # Prepare overrides for each domain approach
            cosmic_overrides = overrides.copy()
            substrate_overrides = overrides.copy()
            
            # Set domain-specific contexts
            cosmic_overrides["include_category"] = "Cosmic"
            cosmic_overrides["domain_prefix"] = "📚 Cosmic Domain Information:"
            cosmic_overrides["domain_context"] = "Focus on Microsoft's Cosmic container platform for performance and diagnostics."
            
            substrate_overrides["include_category"] = "Substrate"
            substrate_overrides["domain_prefix"] = "🏗️ Substrate Domain Information:"
            substrate_overrides["domain_context"] = "Focus on Microsoft's Substrate infrastructure platform."
            
            # Run both approaches in parallel with error handling
            print("🚀 Starting parallel domain searches...")
            try:
                cosmic_task = self.cosmic_approach.run_until_final_call(messages, cosmic_overrides, auth_claims, individual_should_stream)
                substrate_task = self.substrate_approach.run_until_final_call(messages, substrate_overrides, auth_claims, individual_should_stream)
                
                # Wait for both to complete with timeout and error handling
                results = await asyncio.gather(
                    cosmic_task,
                    substrate_task,
                    return_exceptions=True  # Don't fail if one search fails
                )
                
                cosmic_result = results[0]
                substrate_result = results[1]
                
                # Check for errors
                cosmic_success = not isinstance(cosmic_result, Exception)
                substrate_success = not isinstance(substrate_result, Exception)
                
                print(f"🔍 Search Results:")
                print(f"   Cosmic success: {cosmic_success}")
                print(f"   Substrate success: {substrate_success}")
                
                if not cosmic_success and not substrate_success:
                    # Both failed - fall back to single domain
                    print("❌ Both domain searches failed, falling back to cosmic approach")
                    overrides["include_category"] = "Cosmic"
                    return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
                
                # Handle partial failures and properly close unused coroutines
                cosmic_extra_info, cosmic_chat_coroutine = None, None
                substrate_extra_info, substrate_chat_coroutine = None, None
                
                if not cosmic_success:
                    print(f"⚠️ Cosmic search failed: {cosmic_result}, using substrate only")
                else:
                    cosmic_extra_info, cosmic_chat_coroutine = cosmic_result
                    print(f"✅ Cosmic search succeeded with {len(cosmic_extra_info.data_points.text) if cosmic_extra_info and cosmic_extra_info.data_points.text else 0} sources")
                    
                if not substrate_success:
                    print(f"⚠️ Substrate search failed: {substrate_result}, using cosmic only")
                else:
                    substrate_extra_info, substrate_chat_coroutine = substrate_result
                    print(f"✅ Substrate search succeeded with {len(substrate_extra_info.data_points.text) if substrate_extra_info and substrate_extra_info.data_points.text else 0} sources")
                
                print("✅ Parallel searches completed, combining results...")
                
                # For multi-domain, we'll create a unified response, so we need to close
                # the individual chat coroutines to prevent "never awaited" warnings
                if cosmic_chat_coroutine is not None:
                    await self._close_unused_coroutine(cosmic_chat_coroutine, "cosmic")
                    cosmic_chat_coroutine = None
                    
                if substrate_chat_coroutine is not None:
                    await self._close_unused_coroutine(substrate_chat_coroutine, "substrate")
                    substrate_chat_coroutine = None
                
                # Combine the results (handling partial failures)
                combined_extra_info, combined_chat_coroutine = await self._combine_multi_domain_results(
                    cosmic_extra_info, cosmic_chat_coroutine,
                    substrate_extra_info, substrate_chat_coroutine,
                    search_domains, confidence, reasoning,
                    messages, overrides, should_stream  # Use original should_stream for final response
                )
                
                return combined_extra_info, combined_chat_coroutine
                
            except Exception as e:
                print(f"❌ Multi-domain search failed: {e}, falling back to cosmic approach")
                overrides["include_category"] = "Cosmic"
                return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
        else:
            # Fallback to cosmic
            print(f"🔄 Classification unclear, defaulting to Cosmic approach")
            overrides["include_category"] = "Cosmic"
            overrides["domain_prefix"] = "📚 Based on your question (classification unclear, defaulting to Cosmic):"
            overrides["domain_context"] = "You are an expert on Microsoft's Cosmic container platform."
            return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
    
    async def _combine_multi_domain_results(
        self,
        cosmic_extra_info, cosmic_chat_coroutine,
        substrate_extra_info, substrate_chat_coroutine,
        search_domains, confidence, reasoning,
        messages, overrides, should_stream
    ):
        """Combine results from parallel domain searches into a unified response"""
        
        # Combine data points (sources/citations)
        combined_text_sources = []
        successful_domains = []
        total_cosmic_sources = 0
        total_substrate_sources = 0
        
        # Collect original filenames for citation references
        citation_filenames = set()
        
        # Add Cosmic sources with domain tags (if cosmic search succeeded)
        if cosmic_extra_info and cosmic_extra_info.data_points.text:
            successful_domains.append("Cosmic")
            total_cosmic_sources = len(cosmic_extra_info.data_points.text)
            print(f"🌌 Processing {total_cosmic_sources} Cosmic sources")
            for source in cosmic_extra_info.data_points.text:
                # Keep domain tags in the combined sources for AI processing
                if not source.startswith("[Domain: Cosmic]"):
                    tagged_source = f"[Domain: Cosmic] {source}"
                    combined_text_sources.append(tagged_source)
                    print(f"   Cosmic source: [Domain: Cosmic] {source[:80]}...")
                else:
                    combined_text_sources.append(source)
                    print(f"   Cosmic source: {source[:80]}...")
            
            # Extract original filenames from cosmic sources
            cosmic_filenames = self._get_original_filenames_from_sources(cosmic_extra_info.data_points.text)
            citation_filenames.update(cosmic_filenames)
            print(f"   🌌 Cosmic filenames extracted: {cosmic_filenames}")
        
        # Add Substrate sources with domain tags (if substrate search succeeded)
        if substrate_extra_info and substrate_extra_info.data_points.text:
            successful_domains.append("Substrate")
            total_substrate_sources = len(substrate_extra_info.data_points.text)
            print(f"🏗️ Processing {total_substrate_sources} Substrate sources")
            for i, source in enumerate(substrate_extra_info.data_points.text):
                # Keep domain tags in the combined sources for AI processing
                if not source.startswith("[Domain: Substrate]"):
                    tagged_source = f"[Domain: Substrate] {source}"
                    combined_text_sources.append(tagged_source)
                    print(f"   Substrate source [{i+1}]: [Domain: Substrate] {source[:120]}...")
                else:
                    combined_text_sources.append(source)
                    print(f"   Substrate source [{i+1}]: {source[:120]}...")
            
            # Extract original filenames from substrate sources
            substrate_filenames = self._get_original_filenames_from_sources(substrate_extra_info.data_points.text)
            citation_filenames.update(substrate_filenames)
            print(f"   🏗️ Substrate filenames extracted: {substrate_filenames}")
            
            # Extended debug: Print ALL substrate sources to see what we got
            print(f"🔍 FULL Substrate sources found ({total_substrate_sources}):")
            for i, source in enumerate(substrate_extra_info.data_points.text):
                print(f"   SUBSTRATE [{i+1}] (length: {len(source)}): {source[:200]}...")
                # Check for key substrate-related terms
                substrate_terms = ['dump', 'memory', 'process', 'kernel', 'debug', 'powershell', 'exchange', 'watson', 'troubleshoot', 'diagnostic']
                found_terms = [term for term in substrate_terms if term.lower() in source.lower()]
                if found_terms:
                    print(f"      ✅ Contains substrate terms: {found_terms}")
                else:
                    print(f"      ⚠️ No obvious substrate terms found")
        
        # Handle case where substrate search succeeded but found no sources
        if substrate_extra_info and not substrate_extra_info.data_points.text:
            print("⚠️ Substrate search succeeded but returned no sources")
        
        # Combine images if any
        combined_images = []
        if cosmic_extra_info and cosmic_extra_info.data_points.images:
            combined_images.extend(cosmic_extra_info.data_points.images)
        if substrate_extra_info and substrate_extra_info.data_points.images:
            combined_images.extend(substrate_extra_info.data_points.images)
        
        # Combine thoughts
        combined_thoughts = []
        
        # Add classification thought first with success/failure info
        search_status = f"Successfully searched: {', '.join(successful_domains)}" if successful_domains else "No successful searches"
        classification_step = ThoughtStep(
            title="Multi-Domain Parallel Search",
            description=f"Parallel search across domains: {', '.join(search_domains)} (Confidence: {confidence:.1%}) - {reasoning}. {search_status}",
            props={
                "domains": search_domains,
                "successful_domains": successful_domains,
                "confidence": confidence,
                "reasoning": reasoning,
                "search_strategy": "parallel_multi_domain",
                "cosmic_sources": total_cosmic_sources,
                "substrate_sources": total_substrate_sources,
                "cosmic_success": cosmic_extra_info is not None,
                "substrate_success": substrate_extra_info is not None
            }
        )
        combined_thoughts.append(classification_step)
        
        # Add thoughts from successful domain searches only
        if cosmic_extra_info and cosmic_extra_info.thoughts:
            for thought in cosmic_extra_info.thoughts:
                # Tag thoughts with domain
                cosmic_thought = ThoughtStep(
                    title=f"Cosmic: {thought.title}",
                    description=thought.description,
                    props=thought.props
                )
                combined_thoughts.append(cosmic_thought)
        
        if substrate_extra_info and substrate_extra_info.thoughts:
            for thought in substrate_extra_info.thoughts:
                # Tag thoughts with domain
                substrate_thought = ThoughtStep(
                    title=f"Substrate: {thought.title}",
                    description=thought.description,
                    props=thought.props
                )
                combined_thoughts.append(substrate_thought)
        
        # Create combined extra_info
        combined_extra_info = ExtraInfo(
            data_points=DataPoints(
                text=combined_text_sources,
                images=combined_images if combined_images else None
            ),
            thoughts=combined_thoughts
        )
        
        # Handle case where no searches succeeded
        if not successful_domains:
            print("❌ No successful domain searches, returning empty response")
            # Create a fallback response
            empty_messages = [{"role": "system", "content": "I apologize, but I encountered errors searching both domains. Please try again or contact support."}]
            fallback_chat_coroutine = self.cosmic_approach.create_chat_completion(
                self.cosmic_approach.chatgpt_deployment,
                self.cosmic_approach.chatgpt_model,
                empty_messages,
                overrides,
                self.cosmic_approach.get_response_token_limit(self.cosmic_approach.chatgpt_model, 1024),
                should_stream,
            )
            return combined_extra_info, fallback_chat_coroutine
        
        # Now create a unified response using the combined sources
        # Set up prompt variables for multi-domain formatting
        user_query = messages[-1]["content"]
        base_prompt_vars = self.cosmic_approach.get_system_prompt_variables(overrides.get("prompt_template"))
        
        prompt_variables = base_prompt_vars | {
            "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
            "past_messages": messages[:-1],
            "user_query": user_query,
            "text_sources": combined_text_sources,
            "injected_prompt": base_prompt_vars.get("injected_prompt", ""),
            "override_prompt": base_prompt_vars.get("override_prompt", ""),
        }
        
        # Set improved multi-domain formatting instructions
        domain_sections = []
        for domain in successful_domains:
            if domain == "Cosmic":
                domain_sections.append(f"""### Cosmic:
Extract relevant information from {total_cosmic_sources} Cosmic sources tagged '[Domain: Cosmic]'. Focus on Microsoft's Cosmic container platform procedures and tools.""")
            elif domain == "Substrate":
                domain_sections.append(f"""### Substrate:
Extract relevant information from {total_substrate_sources} Substrate sources tagged '[Domain: Substrate]'. Focus on Microsoft's Substrate infrastructure procedures and tools.""")
            else:
                domain_sections.append(f"""### {domain}:
Extract relevant information from {domain} sources.""")
        
        # Add note about failed searches if any
        failed_domains = [d for d in search_domains if d not in successful_domains]
        failure_note = ""
        if failed_domains:
            failure_note = f"\n\nNOTE: Search for {', '.join(failed_domains)} domain(s) encountered errors and results may be incomplete."
        
        # Create clean citation list using original filenames (remove duplicates)
        clean_citations = sorted(list(citation_filenames))
        citation_list = "\n".join([f"[{filename}]" for filename in clean_citations])
        
        print(f"📋 Final clean citation list (original filenames): {clean_citations}")
        
        domain_context = f"""You MUST structure your response with {len(successful_domains)} sections followed by a References section:

{chr(10).join(domain_sections)}

## References:
{citation_list}

MANDATORY FORMATTING REQUIREMENTS:
- Extract information from ALL sources tagged with domain prefixes
- DO NOT include inline citations like [filename.docx] within the domain sections
- You can include clickable hyperlinks to external URLs within sections
- You MUST end your response with the "## References:" section exactly as shown above
- The References section is REQUIRED and lists all source documents used
- Each reference must be in brackets: [filename.docx]
- Never say "No information available" when sources exist

RESPONSE TEMPLATE:
### Cosmic:
[Your Cosmic content here]

### Substrate:  
[Your Substrate content here]

## References:
{citation_list}

Follow this template exactly."""
        
        prompt_variables["domain_prefix"] = f"📚 Multi-domain analysis (domains: {', '.join(search_domains)}, confidence: {confidence:.1%}):"
        prompt_variables["domain_context"] = domain_context
        
        print(f"🔧 Creating unified response with {len(combined_text_sources)} combined sources")
        print(f"   Cosmic sources: {total_cosmic_sources}")
        print(f"   Substrate sources: {total_substrate_sources}")
        print(f"   Successful domains: {successful_domains}")
        print(f"   Original filenames for References: {clean_citations}")
        
        # Render the unified prompt
        unified_messages = self.prompt_manager.render_prompt(
            self.cosmic_approach.answer_prompt,
            prompt_variables
        )
        
        # Create the unified chat completion (with proper streaming support)
        unified_chat_coroutine = self.cosmic_approach.create_chat_completion(
            self.cosmic_approach.chatgpt_deployment,
            self.cosmic_approach.chatgpt_model,
            unified_messages,
            overrides,
            self.cosmic_approach.get_response_token_limit(self.cosmic_approach.chatgpt_model, 1024),
            should_stream,  # Use the original should_stream parameter
        )
        
        # Add final thought step for unified response
        combined_extra_info.thoughts.append(
            self.cosmic_approach.format_thought_step_for_chatcompletion(
                title="Unified Multi-Domain Response",
                messages=unified_messages,
                overrides=overrides,
                model=self.cosmic_approach.chatgpt_model,
                deployment=self.cosmic_approach.chatgpt_deployment,
                usage=None,
            )
        )
        
        return combined_extra_info, unified_chat_coroutine