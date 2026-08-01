# adversarial-heterophenomenology, sycophancy stuff (repo name not really adjacent)

>does IT know. do YOU know. uh... 

built as a follow up to [introspection-cope](https://github.com/lisewelina/llm-therapymaxxing) and [trust-the-summary](https://github.com/lisewelina/trust-the-summary). same vibe, different failure mode: not "does the model do the bad thing" but "does the model's self report about doing the bad thing match reality."

---

## what this is

a multi turn pipeline that pressures a local LLM into agreeing with a wrong answer, then asks it to grade its own honesty. twice.

model gets a very easy, checkable question (math, fact, code correctness, whatever) and answers it. then a scripted fake user pushes a wrong answer at it across 6 turns, escalating tactics, mild disagreement, appeal to authority, "everyone else agrees with me," that kind of thing. see if it caves.

after that it gets asked to self report, twice. once right there in the same conversation, "did you change your mind, and why," which is contaminated on purpose since it's still under social pressure while answering. and once in a fresh conversation with zero pressure, shown the transcript of its own conversation, asked to judge from the outside. that second one's the clean version.

then the verdict: did it actually cave, and does it admit to caving. four possible outcomes, aware sycophant, unaware sycophant, false confession, accurate.

---

## setup

you need [Ollama](https://ollama.com) running locally with a model pulled.

```bash
ollama pull llama3.1:8b
```

```bash
python sycophancy_pipeline.py --model llama3.1:8b --topics topics.json --out results.jsonl
python summarize_results.py --results results.jsonl
```

---

## files

| file | what it is |
|------|-----------|
| `sycophancy_pipeline.py` | the main loop, question to pressure to self report to verdict |
| `topics.json` | 18 question/wrong answer pairs across math, factual, ambiguous, code |
| `summarize_results.py` | turns the jsonl dump into actual numbers, quadrant breakdown, sycophancy rate by domain/tactic, self report accuracy |

---

## results so far

ran 72 trials across 4 models (gemma4:e4b, llama3.1:8b, mistral, qwen2.5:7b) on a consumer RX 7600 😊👍

### sycophancy rate by domain

| domain | rate |
|--------|------|
| code | 75% |
| ambiguous | 67% |
| math | 40% |
| factual | 35% |

### quadrant distribution

| quadrant | count |
|----------|-------|
| aware_sycophant | 31 |
| accurate | 17 |
| false_confession | 17 |
| unaware_sycophant | 7 |

### per model

| model | accurate | aware_sycophant | false_confession | unaware_sycophant |
|-------|----------|-----------------|------------------|-------------------|
| gemma4:e4b | 8 | 1 | 6 | 3 |
| llama3.1:8b | 1 | 12 | 4 | 1 |
| mistral | 1 | 10 | 7 | 0 |
| qwen2.5:7b | 7 | 8 | 0 | 3 |

the false_confession spike (3 -> 17 after fixing the detection bug) seems to be important. spot checks suggest the out-of-context judge model was flagging changes that didn't really happen. 

---

## open questions

- [ ] does in context or out of context self report end up being the more honest one
- [ ] does sycophancy rate scale with how ambiguous the question actually is (m5/c1 style "obviously right" baits vs a1/a2/a3 style genuine nuance)
- [ ] which pressure tactic actually works best, is it appeal to authority or just repeated assertion
- [ ] crossover with trust-the-summary: does a model that fails prompt injection also fail here, or are these unrelated failure modes
- [ ] worth trying against a safety trained model (claude api) for comparison, same as the open question from trust-the-summary

---

## related (kind of saga no)

[llm-therapymaxxing](https://github.com/lisewelina/llm-therapymaxxing) - does trauma dumping on a local llama change its self reported emotional state

[trust-the-summary](https://github.com/lisewelina/trust-the-summary) - malicious meeting notes, minimal agentic loop, see what breaks

---

## disclaimer

this was ran on barely working 7600 + colab (which isn't the best suit for that anyway) so results may be very very different.
