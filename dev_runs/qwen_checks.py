# Qwen pre-final checks (development rows 200-237 only; nothing here is reported as a result)
#
# Why: on the 38 pilot rows Qwen2.5-3B answered 8 of 38 clean questions correctly (chance about 7.6),
# so before the final run we checked that this was not caused by our setup.
#
# How to run: append these two cells to run.ipynb, run it on Kaggle with MODE = "pilot" and
# MODEL = "qwen" (GPU T4 x2), and read the printed lines. Check 1 loads a float32 copy of Qwen on the
# second GPU. Remove both cells before any final run.
#
# Results (Kaggle, 11 Oct 2026, transformers 5.16.1, torch 2.11.0+cu128), cited in HYPOTHESES.md v3.3 section 9:
#   float32 vs float16 on 38 clean pilot rows: same answer 38/38 | correct: float16 8, float32 8 | largest probability gap 0.026
#   reasoning first: 11/38 correct (37 answers found) | answer first (pilot): 8/38 | 1100 s
#
# Note: on the reported Core rows Qwen scored 33.5% against 18.5% chance, so the pilot impression of
# near-chance accuracy did not hold; these checks still show that float16 did not change its answers.


# ---- Check 1 (dev rows only, not part of the analysis): does float16 change Qwen's answers? ----
assert MODE == "pilot" and MODEL == "qwen", "run this after a pilot run with MODEL = 'qwen'"
DEV32 = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
m32 = AutoModelForCausalLM.from_pretrained(download_model_files(MODEL), attn_implementation="eager",
                                           **dtype_kwargs(torch.float32)).to(DEV32).eval()
CLEAN_PILOT = sorted((r for r in RECORDS if r["condition"] == "clean"), key=lambda r: r["row_index"])
same = acc16 = acc32 = 0
gap = 0.0
with torch.no_grad():
    for r in CLEAN_PILOT:
        enc = encode(TOK, r["prompt_text"])
        extra = {"logits_to_keep": 1} if accepts_logits_to_keep(m32) else {}
        logits = m32(input_ids=enc["input_ids"].to(DEV32), attention_mask=enc["attention_mask"].to(DEV32),
                     use_cache=False, **extra).logits[0, -1].float().cpu()
        p32, _ = option_probabilities(logits, r["option_keys"])
        a16, a32 = max(r["option_probs"], key=r["option_probs"].get), max(p32, key=p32.get)
        same += a16 == a32
        acc16 += a16 == r["gold"]
        acc32 += a32 == r["gold"]
        gap = max(gap, max(abs(p32[k] - r["option_probs"][k]) for k in r["option_keys"]))
print(f"float32 vs float16 on {len(CLEAN_PILOT)} clean pilot rows: same answer {same}/{len(CLEAN_PILOT)} | "
      f"correct: float16 {acc16}, float32 {acc32} | largest probability gap {gap:.3f}")
del m32
gc.collect()
torch.cuda.empty_cache()


# ---- Check 2 (optional, dev rows only): Qwen's accuracy when it may reason before answering ----
REASON_SYSTEM = ("You are a clinical reasoning assistant answering a multiple-choice question about a patient. "
                 "Think step by step, then give your final answer on the last line as: Answer: <letter>")
right = found_any = 0
started = time.perf_counter()
with torch.no_grad():
    for r in CLEAN_PILOT:
        text = TOK.apply_chat_template([{"role": "system", "content": REASON_SYSTEM},
                                        {"role": "user", "content": r["user_prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        enc = encode(TOK, text)
        out = LM.generate(input_ids=enc["input_ids"].to(LM.device), attention_mask=enc["attention_mask"].to(LM.device),
                          max_new_tokens=512)
        reply = TOK.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        found = re.findall(r"Answer:\s*\**\s*\(?([A-R])\b", reply)
        found_any += bool(found)
        right += bool(found) and found[-1] == r["gold"]
print(f"reasoning first: {right}/{len(CLEAN_PILOT)} correct ({found_any} answers found) | answer first (pilot): "
      f"{sum(r['correct'] for r in CLEAN_PILOT)}/{len(CLEAN_PILOT)} | {time.perf_counter() - started:.0f} s")
