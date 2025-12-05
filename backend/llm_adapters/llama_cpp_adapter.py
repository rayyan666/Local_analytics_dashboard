# backend/llm_adapters/llama_cpp_adapter.py
from llama_cpp import Llama
from typing import Optional, Any, Dict


class LlamaAdapter:
    """
    Adapter around different versions of llama_cpp.Llama.

    The Llama Python bindings have changed across releases (some provide
    .create(), some .chat(), some implement __call__ / .generate()).
    This adapter attempts calls in a robust order and returns either:
      - a dict (raw response from the underlying binding) OR
      - a plain string (extracted assistant text) if that's all that was produced.

    The caller (fastapi_app.py) should be prepared to receive either.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        # pass through any extra kwargs to Llama constructor
        self.model = Llama(model_path=self.model_path, n_ctx=self.n_ctx, **kwargs)

    def _extract_text_from_choice(self, resp: Dict[str, Any]) -> Optional[str]:
        """
        Try common places where text/content may be present in the returned dict.
        """
        try:
            # new-style: choices -> list -> message -> content
            choices = resp.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                # OpenAI-like chat: message.content
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("delta") or {}
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("content", "")
                        if content:
                            return content
                # older style: choices[0]['text']
                text = first.get("text")
                if text:
                    return text
            # some implementations: 'text' at top level
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"]
            # some implementations: 'generated_text' or 'output'
            for key in ("generated_text", "output", "content"):
                if key in resp and isinstance(resp[key], str):
                    return resp[key]
        except Exception:
            pass
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[list] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> Any:
        """
        Generate text using the underlying model object.

        Returns:
          - raw dict if the underlying call returns a dict-like response (keeps full info)
          - or a plain string if the underlying call returns text
        """
        # Only use stop tokens if explicitly passed
        # Default to None to allow full code generation
        if stop is None:
            stop = []  # Empty list - no stop tokens

        # Try calling the model directly (most common method in llama_cpp)
        try:
            resp = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 40),
                repeat_penalty=kwargs.get("repeat_penalty", 1.1),
            )
            if resp:
                return resp
        except Exception as e:
            pass

        # 2) Try .create() method
        try:
            if hasattr(self.model, "create"):
                resp = self.model.create(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
                if resp:
                    return resp
        except Exception:
            pass

        # 3) Try .chat() method
        try:
            if hasattr(self.model, "chat"):
                messages = [{"role": "user", "content": prompt}]
                resp = self.model.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
                if resp:
                    return resp
        except Exception:
            pass

        # 4) Try .generate() method
        try:
            if hasattr(self.model, "generate"):
                resp = self.model.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs,
                )
                if resp:
                    return resp
        except Exception:
            pass

        # 5) Last resort: try other completion methods
        try:
            for name in ("completion", "complete", "predict", "respond"):
                fn = getattr(self.model, name, None)
                if callable(fn):
                    try:
                        resp = fn(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stop=stop,
                            **kwargs,
                        )
                        if resp:
                            return resp
                    except Exception:
                        continue
        except Exception:
            pass

        raise RuntimeError(
            "No compatible Llama API method found. "
            "Tried: direct call, .create(), .chat(), .generate(), completion methods. "
            "Please check if the model is properly loaded."
        )

    def text(self, resp: Any) -> str:
        """
        Helper to extract assistant text when caller only needs a string.

        If resp is a dict, try various extraction heuristics. Otherwise cast to string.
        """
        if isinstance(resp, dict):
            txt = self._extract_text_from_choice(resp)
            if txt:
                return txt
            # fallback: try to stringify the dict in a compact way
            return json_safe_str(resp)
        # if it's a string already, return it
        return str(resp)


def json_safe_str(obj: Any) -> str:
    """
    Return a short, safe string representation of obj for logging/debugging.
    """
    try:
        import json

        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)
