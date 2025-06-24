import asyncio
import re
import logging
from typing import Any, Optional

from approaches.chatapproach import ChatApproach
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from approaches.approach import ThoughtStep, DataPoints, ExtraInfo, Document


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

    def _clean_source(self, source: str, domain: str) -> str:
        """Remove domain tags from sources to get clean filename: content format"""
        domain_tag = f"[Domain: {domain}] "
        if source.startswith(domain_tag):
            return source[len(domain_tag):]
        return source

    async def _close_unused_coroutine(self, coroutine, name: str):
        """Safely close an unused coroutine to prevent runtime warnings"""
        if coroutine is not None:
            try:
                # Try to cancel first if it's a Task
                if hasattr(coroutine, 'cancel'):
                    coroutine.cancel()
                    try:
                        await coroutine
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling
                # Then close if it's a generator/coroutine
                elif hasattr(coroutine, 'close'):
                    coroutine.close()
            except (GeneratorExit, RuntimeError, asyncio.CancelledError) as e:
                # These are expected when closing coroutines
                logging.debug(f"Expected error closing {name} coroutine: {e}")
            except Exception as e:
                logging.warning(f"Unexpected error closing {name} coroutine: {e}")

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
                overrides["domain_context"] = "You are an expert on Microsoft's Cosmic container platform for performance and diagnostics."
                return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
            elif user_category == "Substrate":
                overrides["domain_prefix"] = "🏗️ Based on your question about Microsoft's Substrate infrastructure:"
                overrides["domain_context"] = "You are an expert on Microsoft's Substrate infrastructure platform."
                return await self.substrate_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
        
        # Extract the latest user question
        user_question = messages[-1]["content"] if messages else ""
        
        # Get domain classification with context
        try:
            domains, confidence, reasoning = await self.domain_classifier.classify_with_context(
                user_question,
                messages[:-1]  # Pass conversation history
            )
            
            logging.info(f"🎯 Orchestrator Classification Result:")
            logging.info(f"   Question: {user_question}")
            logging.info(f"   Domains: {domains}")
            logging.info(f"   Confidence: {confidence:.1%}")
            logging.info(f"   Reasoning: {reasoning}")
            
        except Exception as e:
            logging.error(f"❌ Domain classification error: {e}")
            # Fall back to cosmic domain
            domains = ["Cosmic"]
            confidence = 0.5
            reasoning = f"Classification failed: {str(e)}, defaulting to Cosmic"
        
        # Handle based on classification results
        if len(domains) == 1 and confidence >= 0.6:
            # Single domain with high confidence - no filename extraction needed
            domain = domains[0]
            overrides["include_category"] = domain
            
            # Add domain-specific prompt context
            if domain == "Cosmic":
                overrides["domain_prefix"] = f"📚 Based on your question about Microsoft's Cosmic (confidence: {confidence:.1%}):"
                overrides["domain_context"] = "You are an expert on Microsoft's Cosmic container platform for performance and diagnostics."
            else:
                overrides["domain_prefix"] = f"🏗️ Based on your question about Microsoft's Substrate (confidence: {confidence:.1%}):"
                overrides["domain_context"] = "You are an expert on Microsoft's Substrate infrastructure platform."
            
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
            logging.info(f"🔄 Multi-domain detected, running parallel searches for domains: {domains}")
            
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
            logging.info("🚀 Starting parallel domain searches...")
            try:
                cosmic_task = self.cosmic_approach.run_until_final_call(messages, cosmic_overrides, auth_claims, individual_should_stream)
                substrate_task = self.substrate_approach.run_until_final_call(messages, substrate_overrides, auth_claims, individual_should_stream)
                
                # Wait for both to complete with timeout and error handling
                results = await asyncio.wait_for(
                    asyncio.gather(cosmic_task, substrate_task, return_exceptions=True),
                    timeout=30.0  # 30 second timeout
                )
                
                cosmic_result = results[0]
                substrate_result = results[1]
                
                # Check for errors
                cosmic_success = not isinstance(cosmic_result, Exception)
                substrate_success = not isinstance(substrate_result, Exception)
                
                logging.info(f"🔍 Search Results:")
                logging.info(f"   Cosmic success: {cosmic_success}")
                logging.info(f"   Substrate success: {substrate_success}")
                
                if not cosmic_success and not substrate_success:
                    # Both failed - fall back to single domain
                    logging.error("❌ Both domain searches failed, falling back to cosmic approach")
                    overrides["include_category"] = "Cosmic"
                    return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
                
                # Handle partial failures and properly close unused coroutines
                cosmic_extra_info, cosmic_chat_coroutine = None, None
                substrate_extra_info, substrate_chat_coroutine = None, None
                
                if not cosmic_success:
                    logging.warning(f"⚠️ Cosmic search failed: {cosmic_result}, using substrate only")
                else:
                    cosmic_extra_info, cosmic_chat_coroutine = cosmic_result
                    sources_count = len(cosmic_extra_info.data_points.text) if cosmic_extra_info and cosmic_extra_info.data_points and cosmic_extra_info.data_points.text else 0
                    logging.info(f"✅ Cosmic search succeeded with {sources_count} sources")
                    
                if not substrate_success:
                    logging.warning(f"⚠️ Substrate search failed: {substrate_result}, using cosmic only")
                else:
                    substrate_extra_info, substrate_chat_coroutine = substrate_result
                    sources_count = len(substrate_extra_info.data_points.text) if substrate_extra_info and substrate_extra_info.data_points and substrate_extra_info.data_points.text else 0
                    logging.info(f"✅ Substrate search succeeded with {sources_count} sources")
                
                logging.info("✅ Parallel searches completed, combining results...")
                
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
                logging.error(f"❌ Multi-domain search failed: {e}, falling back to cosmic approach")
                overrides["include_category"] = "Cosmic"
                return await self.cosmic_approach.run_until_final_call(messages, overrides, auth_claims, should_stream)
        else:
            # Fallback to cosmic
            logging.info(f"🔄 Classification unclear, defaulting to Cosmic approach")
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
        
        # Prepare clean source lists (NO domain tags) - like vision approach
        cosmic_sources = []
        substrate_sources = []
        successful_domains = []
        total_cosmic_sources = 0
        total_substrate_sources = 0
        
        # Clean cosmic sources (remove domain tags if present)
        if cosmic_extra_info and cosmic_extra_info.data_points and cosmic_extra_info.data_points.text:
            successful_domains.append("Cosmic")
            total_cosmic_sources = len(cosmic_extra_info.data_points.text)
            logging.info(f"🌌 Processing {total_cosmic_sources} Cosmic sources")
            for source in cosmic_extra_info.data_points.text:
                clean_source = self._clean_source(source, "Cosmic")
                cosmic_sources.append(clean_source)
                logging.debug(f"   Cosmic source: {clean_source[:80]}...")
        
        # Clean substrate sources (remove domain tags if present)
        if substrate_extra_info and substrate_extra_info.data_points and substrate_extra_info.data_points.text:
            successful_domains.append("Substrate")
            total_substrate_sources = len(substrate_extra_info.data_points.text)
            logging.info(f"🏗️ Processing {total_substrate_sources} Substrate sources")
            for i, source in enumerate(substrate_extra_info.data_points.text):
                clean_source = self._clean_source(source, "Substrate")
                substrate_sources.append(clean_source)
                logging.info(f"   ✅ Substrate source [{i+1}]: {clean_source[:120]}...")
        else:
            # DEBUG: Log detailed substrate search results
            logging.warning(f"❌ Substrate sources missing - Debug info:")
            logging.warning(f"   substrate_extra_info exists: {substrate_extra_info is not None}")
            if substrate_extra_info:
                logging.warning(f"   substrate_extra_info.data_points exists: {substrate_extra_info.data_points is not None}")
                if substrate_extra_info.data_points:
                    logging.warning(f"   substrate_extra_info.data_points.text exists: {substrate_extra_info.data_points.text is not None}")
                    if substrate_extra_info.data_points.text:
                        logging.warning(f"   substrate_extra_info.data_points.text length: {len(substrate_extra_info.data_points.text)}")
                        logging.warning(f"   substrate_extra_info.data_points.text content: {substrate_extra_info.data_points.text}")
            logging.warning(f"   Substrate will NOT be in successful_domains")
        
        # Handle case where substrate search succeeded but found no sources
        if substrate_extra_info and substrate_extra_info.data_points and not substrate_extra_info.data_points.text:
            logging.warning("⚠️ Substrate search succeeded but returned no sources")
        
        # Combine images if any
        combined_images = []
        if cosmic_extra_info and cosmic_extra_info.data_points and cosmic_extra_info.data_points.images:
            combined_images.extend(cosmic_extra_info.data_points.images)
        if substrate_extra_info and substrate_extra_info.data_points and substrate_extra_info.data_points.images:
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
        
        # Add substrate thoughts
        if substrate_extra_info and substrate_extra_info.thoughts:
            for thought in substrate_extra_info.thoughts:
                # Tag thoughts with domain
                substrate_thought = ThoughtStep(
                    title=f"Substrate: {thought.title}",
                    description=thought.description,
                    props=thought.props
                )
                combined_thoughts.append(substrate_thought)
        
        # Create combined extra_info with clean sources for citation validation
        # Frontend will get clean sources to validate against
        all_clean_sources = cosmic_sources + substrate_sources
        combined_extra_info = ExtraInfo(
            data_points=DataPoints(
                text=all_clean_sources,  # Provide clean sources for citation validation
                images=combined_images if combined_images else None
            ),
            thoughts=combined_thoughts
        )
        
        # Handle case where no searches succeeded
        if not successful_domains:
            logging.error("❌ No successful domain searches, returning empty response")
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
        
        # Now create a unified response using the separated clean sources
        # Set up prompt variables for multi-domain formatting
        user_query = messages[-1]["content"]
        base_prompt_vars = self.cosmic_approach.get_system_prompt_variables(overrides.get("prompt_template"))
        
        # NEW: Use separate source lists instead of combined tagged sources
        prompt_variables = base_prompt_vars | {
            "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
            "past_messages": messages[:-1],
            "user_query": user_query,
            "cosmic_sources": cosmic_sources,  # Clean cosmic sources
            "substrate_sources": substrate_sources,  # Clean substrate sources
            "text_sources": [],  # Don't use the old combined approach
            "injected_prompt": base_prompt_vars.get("injected_prompt", ""),
            "override_prompt": base_prompt_vars.get("override_prompt", ""),
        }
        
        # Set domain prefix for multi-domain
        prompt_variables["domain_prefix"] = f"📚 Multi-domain analysis (domains: {', '.join(search_domains)}, confidence: {confidence:.1%}):"
        
        logging.info(f"🔧 Creating unified response with separate source lists")
        logging.info(f"   Cosmic sources: {len(cosmic_sources)}")
        logging.info(f"   Substrate sources: {len(substrate_sources)}")
        logging.info(f"   Successful domains: {successful_domains}")
        
        # DEBUG: Log clean sources being sent to AI
        logging.info(f"🔍 Clean Cosmic sources being sent to AI:")
        for idx, source in enumerate(cosmic_sources):
            logging.info(f"   [{idx+1}] {source[:150]}...")
            
        logging.info(f"🔍 Clean Substrate sources being sent to AI:")
        for idx, source in enumerate(substrate_sources):
            logging.info(f"   [{idx+1}] {source[:150]}...")
        
        # Render the unified prompt with separate source lists
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
