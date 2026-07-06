This diagnostic report compares the architectural performance of your **"Baseline Modular Pipeline"** (prior to streaming integration) and your **"Experimental Streaming Pipeline"** (post-async integration).

The performance degradation you are seeing (UPL rising from ~9s to ~91s) is not a flaw in the *idea* of async streaming, but a critical **state-machine synchronization failure** between the LLM output and your orchestrator’s memory.

---

### **Section 5: Architectural Performance Analysis & Diagnostic Findings**

#### **1. Comparative Performance Summary**

| Metric | Baseline (Blocking) | Experimental (Async Streaming) | Variance Analysis |
| --- | --- | --- | --- |
| **UPL (Core Latency)** | ~9.3s | ~91.0s | **10x Degradation** |
| **Response Tokens** | ~99 tokens | ~236 tokens | **2.4x Over-generation** |
| **TTS Performance** | Consistent | Chaotic/Repetitive | **Stream Synchronization Failure** |

#### **2. Diagnostic of Downgraded Behavior**

Your expectation was that async streaming would reduce latency by allowing the system to "hear" the first sentence while the model thinks about the next. However, the system is performing **10 times worse** due to two specific architectural failures:

* **The "Infinite Generation" Hallucination:** In the original blocking code, the model used a standard `decode()` function that automatically respects the "stop token" (`<eos>`). In your new streaming implementation, without a strictly enforced stop criterion in the `TextIteratorStreamer`, the model is not "stopping" when it finishes the clinical answer. It continues to generate filler, whitespace, or redundant tags until it hits the hard `max_new_tokens=256` limit. This is why your `UPL` has jumped to 91 seconds—you are effectively forcing the model to generate 256 tokens every single turn, even if the answer only requires 50.
* **Double-Output Log Pollution:** The orchestrator is double-logging because of a loop-logic conflict. When the async streamer produces tokens, your `orchestrator.py` prints them to the terminal *and* logs them to the CSV. Simultaneously, because the model is hallucinating, it often re-outputs the entire `<Transcription>` and `<Response>` structure at the end of its 256-token limit. Your orchestrator then catches this repeated, hallucinated structure and logs it again. This is why you see the response noted twice in your CSV logs.

#### **3. Linguistic Misalignment (The Language Tag Issue)**

You noted that the system frequently defaults to English even for Hindi inputs.

* **The Root Cause:** The language-tagging logic in the `_format_prompt` function is sensitive to the **History Window**. In the "New" implementation, because the generation process is hallucinating and outputting repeated text, your `history` array is being polluted with hundreds of English tokens. These English tokens dominate the context window, causing the model to "forget" the current Hindi turn and default to its dominant training language (English).

---

### **4. Resolution Strategy**

#### **A. Implementable Fix (Direct Engineering Solves)**

1. **Hard-Stop Enforcement:** In your `gemma_engine.py`, you **must** pass the `stopping_criteria` to the model. Do not rely on the streamer to stop; you must force the model to shut off the moment it generates the `</Response>` tag.
2. **Clean Token Flushing:** Modify the orchestrator’s `run()` loop to strictly discard any token generated *after* the `</Response>` tag is detected. If the model emits anything after that tag, the orchestrator should immediately kill the streamer thread.

#### **B. Research Problem (State Management)**

* **Multimodal Attention Drift:** The core research problem here is **"Contextual Anchor Decay."** When using multimodal models, the sheer number of tokens involved in audio-to-text processing (100k+ features) combined with historical text tokens causes the model's attention mechanism to "drift."
* **The Study:** We need to research whether the model requires a "System-Message Reset" after every turn, or if we can implement **Dynamic Prompting** where the model is periodically reminded of the target language through a lightweight classifier *before* the prompt is generated, rather than relying on the history buffer.

---

### **Recommendation for Next Steps**

The streaming parser logic in your `orchestrator.py` is currently too loose. It is accepting hallucinations as valid response data.

**I strongly recommend we return to your Original code for 24 hours to stabilize your data collection.** Once your baseline is stable again, we will implement a "strict boundary" streaming parser that explicitly kills the LLM process the moment it finishes the `<Response>` tag. Do you want me to provide that "Hard-Stop" streaming version, or would you prefer to stay on the stable blocking code for now to finish your report?
