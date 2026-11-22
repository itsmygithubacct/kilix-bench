"""lm-eval GGUF completions that retain Python indentation."""

from lm_eval.api.registry import register_model
from lm_eval.models.gguf import GGUFLM
from lm_eval.models.utils import normalize_gen_kwargs


@register_model("kilix-gguf-code")
class GGUFCode(GGUFLM):
    def _generate_one(self, args):
        prompt, generation = args
        options = normalize_gen_kwargs(generation, self.max_gen_toks)
        response = self.gguf_completion(context=prompt, stop=options["until"],
                                        max_tokens=options["max_gen_toks"])
        if not response or not response.get("choices"):
            raise RuntimeError("The GGUF server returned no completion.")
        return response["choices"][0]["text"]
