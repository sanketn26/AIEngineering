# Gate 6 (stretch) — make the model pick, not write

> When the answer is one of a few known options, don't make the model write. Make it pick.
> Picking is fast enough to run while the user waits.

A normal LLM call asks the model to write: `"This ticket is about billing because..."`. Your code then reads the text and hunts for the answer. That takes about a second.

But the support queue only has five answers: billing, shipping, account, product, other. So give the model the list, and ask it to point at one. It reads the ticket once and scores each option. No writing, no parsing. About a tenth of a second.

```text
Writing:   ticket ──▶ model writes 30 words ──▶ your code finds "billing"      ~1 s
Picking:   ticket + 5 options ──▶ model scores each option ──▶ billing 0.91    ~0.1 s
```

You also get something writing never gives you: **how sure it is.** `billing 0.91` and `billing 0.46, shipping 0.44` pick the same answer, but only the first one should run without a human.

## The capstone in five steps

1. **Pick a real-time moment.** A customer waiting on a ticket, a chat message about to post, a payment about to clear.
2. **List the answers,** and include `other` for "none of these / not sure". Without it the model must pick a real option even for nonsense.
3. **Make the model pick, and time it.** Then time the same model writing. Same laptop, same tickets.
4. **Check that it picks well.** Reverse the order of the options. If the answer changes, the model is choosing by letter, not by meaning.
5. **Decide what happens when it's unsure.** Automate, send to a bigger model, or send to a human. That rule lives in your code, not in the prompt.

**Done when you can say:** *"My decision takes X ms, is right Y% of the time, and when it's unsure, Z happens."*

```bash
pytest tests/test_decision.py -v                    # no model needed
python -m decision.calibration                      # accuracy, and "when it's sure, is it right?"

pip install -r requirements-decision.txt            # real model: Mac, Linux, or Windows
python -m decision.calibration --backend transformers
python -m decision.bench                            # picking vs writing, timed

pip install -r requirements-decision-mlx.txt        # Apple silicon only, separate environment
python -m decision.bench --backend mlx
```

## What we measured

Qwen2.5-0.5B-Instruct on an Apple M-series laptop, 100 tickets. Yours will differ; measure your own.

| | Picking | Writing |
|---|---|---|
| Time per decision | **~120 ms** | ~1,000 ms |
| Right answer | 52% | 36–44% |

Faster *and* more accurate on a small model. Two surprises are worth the capstone on their own:

- **The model liked the letter B.** It picked "shipping" (option B) for 16 of 25 tickets. Reverse the list and it picked "product" (now option B) 20 times. A confident answer can be a habit, not a judgement.
- **Bigger models have habits too.** The 1.5B model leaned toward option A instead. Just reversing the list moved it from 72% to 88% right. Same model, same tickets, different order.
- **Bigger models change the trade-off.** On a 1.5B model, a single pick was right 72% of the time and writing was right 84%. Picking was still 6× faster. Fixing the letter habit (next section) lifted picking to 88%, at most of the speed cost.

## Where it helps

These businesses make **the same kind of decision thousands of times a day**, from a fixed set of outcomes, while a customer or a process waits.

| Business | Scenario | The decision |
|---|---|---|
| **E-commerce / retail** | Customer opens a support chat | Which team: returns, delivery, payment, product question? |
| | Product review submitted | Publish, hold for review, reject |
| | Return request | Auto-approve, ask for photos, send to an agent |
| **Banking / fintech** | Card payment in progress | Approve, step-up verification, decline |
| | Customer message in the app | Complaint, fraud report, general query (complaints often have legal deadlines) |
| | Loan application arrives | Fast-track, standard review, needs documents |
| **Insurance** | New claim submitted | Simple and auto-payable, needs an adjuster, possible fraud |
| **Telecom / utilities** | Customer calls the voice line | Billing, outage, upgrade, cancel (the caller hears silence while it decides) |
| | Customer says "I want to leave" | Retention offer, process cancellation, escalate |
| **Healthcare (admin side)** | Patient portal message | Urgent to a nurse, appointment, prescription refill, billing |
| **HR / recruiting** | Application received | Meets minimum requirements, doesn't, borderline |
| **Social / marketplaces / gaming** | Chat message or listing posted | Allow, hide, review (millions a day, before anyone sees it) |
| **SaaS / IT** | Help-desk ticket or alert fires | Which team, and how urgent |
| **Legal / compliance** | Email or document scanned | Contains personal data or not; privileged or not |
| **Logistics** | Delivery exception reported | Reschedule, refund, investigate |

**What they share:** high volume, a short fixed list of outcomes, and a cost to waiting: a customer on hold, a payment timing out, content going live. The confidence score matters to the business too. Sure cases run automatically, unsure ones go to a person, so people spend their time where judgement is actually needed.

## Where it doesn't

| Business | Scenario | Why not |
|---|---|---|
| **Marketing / content** | Writing product descriptions, ads, emails | The output *is* the text; there's nothing to pick from |
| **Customer service (reply)** | Drafting the actual answer to the customer | Routing the ticket fits; writing the reply doesn't |
| **Legal** | Reviewing a contract, summarising a case | Needs reasoning and written explanation |
| **Finance ops** | Reading invoices: amounts, dates, suppliers | Extracting values, not choosing from a list |
| **Healthcare (clinical)** | Diagnosis or treatment decisions | Regulated; a bare score without reasoning isn't acceptable, and the options aren't a short fixed list |
| **Regulated credit decisions** | Rejecting a loan | Many regulators require a stated reason; a score alone isn't enough |
| **Research / analytics** | "Why did sales drop last quarter?" | Open question, no fixed set of answers |
| **Retail catalogues** | Matching a product to one of 50,000 categories | Too many options for one pick; narrow the list first with search |
| **Back office** | Monthly reporting, overnight batch jobs | Nobody is waiting, so the speed gain doesn't matter |
| **Hosted-API-only teams** | Any of the above | Most hosted APIs (and Ollama) don't expose per-option scores; you need to run the model yourself |

## The quick test

It fits if the business can answer yes to all three:

1. Can we write down every possible answer on one page?
2. Is someone, or something, waiting for the answer?
3. Do we make this decision thousands of times?

A **no to the first** means the model needs to write. A **no to the second or third** means picking still works, but the speed gain isn't worth the extra setup.

And one limit even when it fits: **a high score is not a promise.** In our run, a "my account was hacked" ticket went to `account` with 0.92 confidence. Rules for the `other` option and a separate security check catch what a confident model misses.

## Going deeper

- **Fixing the letter habit.** `--rotations 5` asks the same question with the options shuffled five ways and averages the results, so no option benefits from its letter. It raised the 0.5B model from 52% to 72%, but costs five passes.
- **"When it says it's sure, is it right?"** `decision.calibration` prints a table of confidence against actual accuracy, plus a single number for the gap (ECE). Use it to set the "automate above this score" line from data.
- **How the pick works.** The prompt ends at `Label:`. The model computes a score for every word it knows as the next word. `scoring.py` keeps only the scores for `A`–`E` and turns those five into percentages. `resolve_label_ids` checks that each letter really is a single token after the prompt, and refuses to run if not.
- **Other tools.** Serving engines such as vLLM and SGLang expose the same scores behind an API, which adds batching for many users. The idea is identical.

| File | What's in it |
|---|---|
| `scoring.py` | The prompt, the pick, and three backends: mock, Transformers, MLX |
| `policy.py` | The "automate, bigger model, or human" rule |
| `calibration.py` | "When it's sure, is it right?" |
| `bench.py` | Picking vs writing, timed on the same model |
