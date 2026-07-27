import os


class LLMService:
    """
    Handles connections and request formatting for Large Language Models (LLMs).
    Supports providers:
    1. Groq (using groq SDK)
    2. OpenAI (using openai SDK)
    3. OpenRouter (using openai SDK with a custom base URL)
    """
    def __init__(self):
        """
        Initializes the service and automatically configures the appropriate client
        based on available environment variables.
        """
        self.provider = "none"
        self.client = None
        self.model = None
        self._configure()

    def _configure(self):
        """
        Detects API keys in environment variables to set up the client connection.
        Prioritizes Groq first, then OpenRouter, then standard OpenAI.
        """
        # 1. Check for Groq API configuration
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            from groq import Groq

            self.provider = "groq"
            self.client = Groq(api_key=groq_key)
            self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            self.vision_model = os.getenv("GROQ_VISION_MODEL", "").strip()
            return

        # 1.5. Check for NVIDIA API configuration
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_key:
            from openai import OpenAI

            self.provider = "nvidia"
            # NVIDIA Nim uses standard OpenAI library pointing to its custom URL
            self.client = OpenAI(
                api_key=nvidia_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
            self.model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
            self.vision_model = os.getenv("NVIDIA_VISION_MODEL", "").strip()
            return

        # 2. Check for OpenRouter or OpenAI API configuration
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openrouter_key or openai_key:
            from openai import OpenAI

            self.provider = "openrouter" if openrouter_key else "openai"
            self.client = OpenAI(
                api_key=openrouter_key or openai_key,
                # OpenRouter requires a custom API url endpoint, OpenAI uses default
                base_url=(
                    os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
                    if openrouter_key
                    else None
                ),
            )
            # Pick a default model depending on provider
            self.model = os.getenv(
                "OPENAI_MODEL",
                "openai/gpt-4o-mini" if openrouter_key else "gpt-4o-mini",
            )
            # Use specific vision model or fallback to primary text model
            self.vision_model = os.getenv("VISION_MODEL", self.model).strip()

    @property
    def available(self):
        """
        Returns True if a valid client has been initialized, False otherwise.
        """
        return self.client is not None

    @property
    def vision_available(self):
        """
        Returns True if a client is active and a visual/multimodal model name is specified.
        """
        return self.available and bool(getattr(self, "vision_model", ""))

    def complete(self, messages, max_tokens=900):
        """
        Generates a standard text completion response.
        - `messages`: List of dicts in role-content format (e.g. [{"role": "user", "content": "hi"}])
        - `max_tokens`: Maximum response token length.
        """
        if not self.available:
            return None

        # Call the chat completions endpoint with low temperature for deterministic answers
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.15,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def complete_with_images(self, system_prompt, question, context, images, max_tokens=900):
        """
        Generates a multimodal text response by sending text context and image inputs.
        - `system_prompt`: Instructions defining role behavior (answer rules).
        - `question`: User's input question.
        - `context`: Text document chunks.
        - `images`: List of base64 data URL images.
        """
        if not self.vision_available:
            return None

        # Build the user message content structure
        content = [
            {
                "type": "text",
                "text": (
                    f"Question: {question}\n\nRetrieved text excerpts:\n{context}\n\n"
                    "Inspect the attached PDF page images, including charts, tables and figures. "
                    "Use only this material and cite pages as [Page X]."
                ),
            }
        ]
        
        # Append base64 image data URLs to the user prompt payload
        for image in images:
            content.append(
                {
                    "type": "text",
                    "text": f"PDF page {image['page']}:",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image["data_url"]},
                }
            )

        # Call the chat completions API using the configured vision model
        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
