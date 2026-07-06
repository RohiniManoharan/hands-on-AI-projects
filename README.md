**Multiprovider_LLM_fallback_withPydantic.py**---- demonstrates a multi-provider LLM fallback system using OpenAI, Gemini, and Groq, with Pydantic used to define and validate structured model outputs. It shows how to route requests across providers for reliability while keeping responses consistent and schema-based.
**API_errors_handrolled_retry.py  & API_errors_Tenacity.py**
This project demonstrates retry mechanisms and error handling strategies while working with API calls, especially LLM APIs.

- Simulation of retryable and non-retryable API errors
- Implementation of exponential backoff with jitter
- Use of Tenacity library for robust retry handling
- Fallback mechanism using Groq LLM API
- Comparison between manual retry logic and library-based retries


