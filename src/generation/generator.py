"""Answer generation with [n] citations.

Backends (config generation.backend):
  hf_causal   instruction-tuned decoder model, e.g. Qwen/Qwen2.5-1.5B-Instruct (best citations)
  hf_seq2seq  e.g. google/flan-t5-base (small and fast, but often skips citations)
  anthropic   Claude via API (needs ANTHROPIC_API_KEY)
"""
import os

from src.utils.helpers import load_config

RULES = (
    "Answer the question using ONLY the numbered passages below. "
    "After each claim, cite the passage number in square brackets, like [1]. "
    "If the passages do not contain the answer, reply exactly: I don't know."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(f"[{p['rank']}] {p['text']}" for p in passages)
    return f"{RULES}\n\nPassages:\n{context}\n\nQuestion: {question}\nAnswer:"


class Generator:
    def __init__(self, backend: str, model: str, max_new_tokens: int = 128,
                 max_passages: int = 3, anthropic_model: str = ""):
        self.backend, self.model_name = backend, model
        self.max_new_tokens, self.max_passages = max_new_tokens, max_passages
        self.anthropic_model = anthropic_model
        self._tok = self._model = self._client = None
        self.device = "cpu"

    @classmethod
    def from_config(cls) -> "Generator":
        g = load_config()["generation"]
        return cls(g["backend"], g["model"], g["max_new_tokens"], g["max_passages"],
                   g.get("anthropic_model", ""))

    def generate(self, question: str, passages: list[dict]) -> str:
        prompt = build_prompt(question, passages[: self.max_passages])
        if self.backend == "hf_causal":
            return self._causal(prompt)
        if self.backend == "hf_seq2seq":
            return self._seq2seq(prompt)
        if self.backend == "anthropic":
            return self._anthropic(prompt)
        raise ValueError(f"Unknown generation.backend: {self.backend}")

    # ---- Hugging Face
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        auto = AutoModelForSeq2SeqLM if self.backend == "hf_seq2seq" else AutoModelForCausalLM
        self._model = auto.from_pretrained(self.model_name).to(self.device).eval()

    def _seq2seq(self, prompt: str) -> str:
        import torch
        self._load()
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()

    def _causal(self, prompt: str) -> str:
        import torch
        self._load()
        if getattr(self._tok, "chat_template", None):
            prompt = self._tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---- Anthropic API
    def _anthropic(self, prompt: str) -> str:
        if self._client is None:
            from anthropic import Anthropic
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to use this backend.")
            self._client = Anthropic()
        msg = self._client.messages.create(model=self.anthropic_model, max_tokens=self.max_new_tokens,
                                           temperature=0.0, messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
