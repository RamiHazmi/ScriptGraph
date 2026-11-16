"""
LLM client with multi-provider fallback (OpenRouter + Groq).
"""

from openai import OpenAI, APIStatusError
import config


class LLMClient:
    """
    LLM client with automatic multi-provider fallback.
    
    Supports:
    - OpenRouter (paid models + free models)
    - Groq (FREE, super fast)
    
    Falls through providers automatically on rate limits/errors.
    """
    
    def __init__(self):
        # Initialize OpenRouter client
        self.openrouter_client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Initialize Groq client (if API key provided)
        self.groq_client = None
        if hasattr(config, 'GROQ_API_KEY') and config.GROQ_API_KEY and not config.GROQ_API_KEY.startswith("gsk_your_"):
            try:
                self.groq_client = OpenAI(
                    api_key=config.GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
            except:
                print("⚠️  Groq API key invalid, will skip Groq models")
        
        # Provider chain: [(provider, model_name), ...]
        self.providers = config.LLM_PROVIDERS
        self.provider_index = 0
        self.current_provider, self.current_model = self.providers[self.provider_index]
        
        # Count available providers
        groq_count = sum(1 for p, _ in self.providers if p == "groq")
        openrouter_count = sum(1 for p, _ in self.providers if p == "openrouter")
        
        print(f"🤖 LLM Client initialized with {len(self.providers)} models")
        print(f"   OpenRouter: {openrouter_count} models")
        print(f"   Groq: {groq_count} models {'(FREE, fast)' if groq_count > 0 else '(no API key)'}")
        print(f"   Primary: {self.current_provider}/{self.current_model}")
    
    def _get_client(self, provider):
        """Get the appropriate client for the provider."""
        if provider == "openrouter":
            return self.openrouter_client
        elif provider == "groq":
            if self.groq_client is None:
                raise Exception("Groq client not initialized (missing API key)")
            return self.groq_client
        else:
            raise Exception(f"Unknown provider: {provider}")
    
    def chat(self, messages, temperature=0, max_tokens=4000):
        """
        Make LLM API call with automatic multi-provider fallback.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            OpenAI response object
        """
        attempts = 0
        max_attempts = len(self.providers)
        
        while attempts < max_attempts:
            try:
                client = self._get_client(self.current_provider)
                
                response = client.chat.completions.create(
                    model=self.current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response
                
            except APIStatusError as e:
                error_code = e.status_code
                error_body = getattr(e, 'body', {})
                
                # Check if it's a credit/rate limit issue
                if error_code in [402, 429, 503]:
                    print(f"\n⚠️  {self.current_provider}/{self.current_model} exhausted ({error_code})")
                    
                    # Try next provider in chain
                    if self.provider_index < len(self.providers) - 1:
                        self.provider_index += 1
                        self.current_provider, self.current_model = self.providers[self.provider_index]
                        attempts += 1
                        
                        is_free = self.current_provider == "groq" or ":free" in self.current_model
                        free_tag = " [FREE]" if is_free else ""
                        print(f"🔄 Switching to fallback #{self.provider_index + 1}{free_tag}: {self.current_provider}/{self.current_model}")
                        
                        continue  # Retry with next provider
                    else:
                        print(f"\n❌ All {len(self.providers)} providers exhausted!")
                        print("💡 Solutions:")
                        print("   1. Wait a few minutes and retry")
                        print("   2. Add Groq API key (FREE): https://console.groq.com")
                        print("   3. Add more FREE models to config.py")
                        raise Exception("All LLM providers exhausted")
                
                elif error_code == 404:
                    print(f"\n⚠️  Model '{self.current_model}' not found on {self.current_provider}")
                    
                    # Try next provider
                    if self.provider_index < len(self.providers) - 1:
                        self.provider_index += 1
                        self.current_provider, self.current_model = self.providers[self.provider_index]
                        attempts += 1
                        print(f"🔄 Trying next: {self.current_provider}/{self.current_model}")
                        continue
                    else:
                        raise
                
                else:
                    # Other API error
                    print(f"\n❌ API Error {error_code}: {error_body}")
                    raise
            
            except Exception as e:
                error_msg = str(e)
                
                # Skip Groq if API key missing
                if "Groq client not initialized" in error_msg:
                    print(f"\n⚠️  Groq unavailable (no API key), skipping...")
                    if self.provider_index < len(self.providers) - 1:
                        self.provider_index += 1
                        self.current_provider, self.current_model = self.providers[self.provider_index]
                        attempts += 1
                        continue
                    else:
                        raise
                
                print(f"\n❌ Unexpected error with {self.current_provider}/{self.current_model}: {e}")
                
                # Try next provider on unexpected errors
                if self.provider_index < len(self.providers) - 1:
                    self.provider_index += 1
                    self.current_provider, self.current_model = self.providers[self.provider_index]
                    attempts += 1
                    print(f"🔄 Trying next: {self.current_provider}/{self.current_model}")
                    continue
                else:
                    raise
    
    def get_model_name(self):
        """Get current model name."""
        return f"{self.current_provider}/{self.current_model}"
    
    def reset_to_primary(self):
        """Reset to primary provider (useful for new sessions)."""
        self.provider_index = 0
        self.current_provider, self.current_model = self.providers[0]
        print(f"🔄 Reset to primary: {self.current_provider}/{self.current_model}")
    
    def get_fallback_status(self):
        """Get current fallback status."""
        return {
            "current_provider": self.current_provider,
            "current_model": self.current_model,
            "provider_index": self.provider_index + 1,
            "total_providers": len(self.providers),
            "remaining_fallbacks": len(self.providers) - self.provider_index - 1
        }
